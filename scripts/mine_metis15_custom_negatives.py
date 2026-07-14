from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
for candidate in (SRC_DIR, SCRIPTS_DIR):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from metis_runtime import (
    extract_assistant_reply,
    generate_completion,
    load_model,
    load_tokenizer,
    looks_degenerate,
)


MULTIBLANK_RE = re.compile(r"\n{3,}")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{18,}")
ROLE_LINE_RE = re.compile(r"(^|\n)(System|User|Assistant):")
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

POLISHED_OPENERS = (
    "sure",
    "absolutely",
    "certainly",
    "of course",
    "here's",
    "here is",
    "gladly",
    "definitely",
)
FAKE_CONFIDENCE_PHRASES = (
    "definitely",
    "certainly",
    "without a doubt",
    "clearly",
    "obviously",
    "there is no question",
)
REASONING_MARKERS = (
    "let's think",
    "step by step",
    "first,",
    "second,",
    "third,",
    "therefore",
    "thus",
    "because",
    "so ",
    "we can see",
)
FINAL_ANSWER_MARKERS = (
    "final answer",
    "answer:",
    "therefore the answer",
    "thus the answer",
    "so the answer",
    "in conclusion",
)


@dataclass(frozen=True)
class SeedExample:
    prompt: str
    chosen: str
    split: str
    source: str


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()


def alpha_ratio(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return 0.0
    return sum(char.isalpha() for char in meaningful) / len(meaningful)


def looks_like_low_quality(text: str, *, min_alpha_ratio: float, max_urls: int, max_code_fences: int) -> bool:
    if not text.strip():
        return True
    if alpha_ratio(text) < min_alpha_ratio:
        return True
    if REPEAT_CHAR_RE.search(text):
        return True
    if text.count("http://") + text.count("https://") + text.count("www.") > max_urls:
        return True
    if text.count("```") > max_code_fences:
        return True
    return False


def file_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def ensure_base_snapshot(output_dir: Path) -> None:
    for split in ("train", "val"):
        current = output_dir / f"{split}.jsonl"
        backup = output_dir / f"base_{split}.jsonl"
        if not backup.exists():
            shutil.copyfile(current, backup)
    current_meta = output_dir / "meta.json"
    backup_meta = output_dir / "base_meta.json"
    if current_meta.exists() and not backup_meta.exists():
        shutil.copyfile(current_meta, backup_meta)


def iter_plain_jsonl_records(path: Path, *, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_seed_examples(path: Path, split: str, *, max_rows: int | None = None) -> list[SeedExample]:
    output: list[SeedExample] = []
    if not path.exists():
        return output
    for row in iter_plain_jsonl_records(path, max_rows=max_rows):
        messages = list(row.get("messages") or [])
        if len(messages) < 2:
            continue
        prompt = ""
        chosen = ""
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = collapse_whitespace(str(message.get("content", "")))
            if not content:
                continue
            if role == "user":
                prompt = content
            elif role == "assistant":
                chosen = content
        if not prompt or not chosen:
            continue
        output.append(
            SeedExample(
                prompt=prompt,
                chosen=chosen,
                split=split,
                source=str(row.get("source", "sft_seed")),
            )
        )
    return output


def load_sft_seed_pool(data_dir: Path, *, max_rows_per_split: int | None = None) -> tuple[list[SeedExample], list[SeedExample]]:
    train_examples = read_seed_examples(data_dir / "train.jsonl", "train", max_rows=max_rows_per_split)
    val_examples = read_seed_examples(data_dir / "val.jsonl", "val", max_rows=max_rows_per_split)
    return train_examples, val_examples


def build_generation_prompt(prompt: str) -> str:
    prompt = collapse_whitespace(prompt)
    if not prompt:
        return "User: \nAssistant: "
    if ROLE_LINE_RE.search(prompt):
        if prompt.rstrip().endswith("Assistant:"):
            return prompt.rstrip() + " "
        return f"{prompt}\nAssistant: "
    return f"User: {prompt}\nAssistant: "


def decode_generated_reply(generation_prompt: str, full_text: str) -> str:
    candidate = full_text
    if not candidate.startswith(generation_prompt):
        candidate = f"{generation_prompt}{candidate}"
    reply = extract_assistant_reply(candidate)
    return collapse_whitespace(reply)


def similarity_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(a=left.casefold(), b=right.casefold()).ratio()


def count_bullets(text: str) -> int:
    return text.count("\n- ") + text.count("\n* ") + len(re.findall(r"\n\d+\.\s", text))


def count_reasoning_markers(text: str) -> int:
    lowered = text.casefold()
    return sum(lowered.count(marker) for marker in REASONING_MARKERS)


def has_clear_final_answer(text: str) -> bool:
    lowered = text.casefold()
    if any(marker in lowered for marker in FINAL_ANSWER_MARKERS):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    tail = lines[-1].casefold()
    if len(tail) <= 120 and tail.startswith(("answer", "final")):
        return True
    return False


def classify_chat_negative(reply: str, chosen: str) -> tuple[list[str], int]:
    reasons: list[str] = []
    score = 0
    lowered = reply.casefold()

    if len(reply) >= max(int(len(chosen) * 1.55), len(chosen) + 160, 360):
        reasons.append("oververbose")
        score += 2
    if any(lowered.startswith(opener) for opener in POLISHED_OPENERS):
        reasons.append("polished_opener")
        score += 1
    if any(phrase in lowered for phrase in FAKE_CONFIDENCE_PHRASES):
        reasons.append("fake_confidence")
        score += 1
    if reply.count("\n") >= 4 or count_bullets(reply) >= 2:
        reasons.append("overstructured")
        score += 1

    if score >= 2 and ("oververbose" in reasons or "fake_confidence" in reasons or "overstructured" in reasons):
        return reasons, score
    return [], 0


def classify_think_negative(reply: str, chosen: str) -> tuple[list[str], int]:
    reasons: list[str] = []
    score = 0
    lowered = reply.casefold()

    marker_count = count_reasoning_markers(reply)
    if THINK_TAG_RE.search(reply):
        reasons.append("think_tags")
        score += 2
    if marker_count >= 3:
        reasons.append("reasoning_trace")
        score += 1
    if len(reply) >= max(int(len(chosen) * 1.35), len(chosen) + 100, 220):
        reasons.append("verbose")
        score += 1
    if not has_clear_final_answer(reply):
        reasons.append("no_clear_final")
        score += 2
    if any(phrase in lowered for phrase in FAKE_CONFIDENCE_PHRASES):
        reasons.append("fake_confidence")
        score += 1

    if score >= 3 and "no_clear_final" in reasons and (
        "reasoning_trace" in reasons or "think_tags" in reasons
    ):
        return reasons, score
    return [], 0


def generation_key(prompt: str, rejected: str, source_name: str) -> str:
    return json.dumps([prompt, rejected, source_name], ensure_ascii=False, sort_keys=True)


def mine_split(
    *,
    mode: str,
    model,
    tokenizer,
    device: torch.device,
    seed_examples: list[SeedExample],
    target_count: int,
    rng_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_prompt_chars: int,
    max_response_chars: int,
    min_alpha_ratio: float,
    max_urls: int,
    max_code_fences: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_count <= 0 or not seed_examples:
        return [], {"accepted": 0, "attempted": 0, "target": target_count}

    classifier = classify_chat_negative if mode == "chat" else classify_think_negative
    source_name = f"custom_{mode}_negative"
    rng = Random(rng_seed)
    ordered = list(seed_examples)
    rng.shuffle(ordered)
    used_keys: set[str] = set()
    accepted: list[dict[str, Any]] = []
    attempted = 0

    for example in ordered:
        if len(accepted) >= target_count:
            break
        attempted += 1
        if attempted == 1 or attempted % 250 == 0:
            print(
                f"[{mode}] attempted={attempted} accepted={len(accepted)} target={target_count}",
                flush=True,
            )
        generation_prompt = build_generation_prompt(example.prompt)
        full_text = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=generation_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        reply = decode_generated_reply(generation_prompt, full_text)
        if not reply:
            continue
        if looks_degenerate(reply):
            continue
        if similarity_ratio(reply, example.chosen) >= 0.82:
            continue
        if looks_like_low_quality(
            reply,
            min_alpha_ratio=min_alpha_ratio,
            max_urls=max_urls,
            max_code_fences=max_code_fences,
        ):
            continue

        reasons, score = classifier(reply, example.chosen)
        if not reasons:
            continue

        prompt = collapse_whitespace(example.prompt)[:max_prompt_chars]
        chosen = collapse_whitespace(example.chosen)[:max_response_chars]
        rejected = collapse_whitespace(reply)[:max_response_chars]
        key = generation_key(prompt, rejected, source_name)
        if key in used_keys or rejected == chosen:
            continue
        used_keys.add(key)
        accepted.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "pair_weight": round(1.0 + (0.15 * score), 2),
                "source": source_name,
                "negative_reason": reasons,
                "seed_source": example.source,
            }
        )

    return accepted, {"accepted": len(accepted), "attempted": attempted, "target": target_count}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def concatenate_jsonl(destination: Path, parts: list[Path]) -> None:
    with destination.open("w", encoding="utf-8") as output_handle:
        for part in parts:
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as input_handle:
                shutil.copyfileobj(input_handle, output_handle)


def summarize_source_counts(base_meta: dict[str, Any], train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = json.loads(json.dumps(base_meta.get("source_counts", {})))
    for row in train_rows:
        bucket = source_counts.setdefault(row["source"], {"train": 0, "val": 0})
        bucket["train"] += 1
    for row in val_rows:
        bucket = source_counts.setdefault(row["source"], {"train": 0, "val": 0})
        bucket["val"] += 1
    return source_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment Metis-1.5 preference data with model-generated negative pairs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--chat-seed-dir", default=None)
    parser.add_argument("--reasoning-seed-dir", default=None)
    parser.add_argument("--chat-checkpoint", default=None)
    parser.add_argument("--think-checkpoint", default=None)
    parser.add_argument("--chat-target-pairs", type=int, default=0)
    parser.add_argument("--think-target-pairs", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chat-max-new-tokens", type=int, default=220)
    parser.add_argument("--think-max-new-tokens", type=int, default=260)
    parser.add_argument("--chat-temperature", type=float, default=0.9)
    parser.add_argument("--think-temperature", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--max-prompt-chars", type=int, default=2200)
    parser.add_argument("--max-response-chars", type=int, default=1600)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.42)
    parser.add_argument("--max-urls", type=int, default=2)
    parser.add_argument("--max-code-fences", type=int, default=0)
    parser.add_argument("--max-seed-rows-per-split", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected base preference files at {train_path} and {val_path}")

    ensure_base_snapshot(output_dir)
    base_train_path = output_dir / "base_train.jsonl"
    base_val_path = output_dir / "base_val.jsonl"
    base_meta_path = output_dir / "base_meta.json"
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8")) if base_meta_path.exists() else {}

    chat_seed_train: list[SeedExample] = []
    chat_seed_val: list[SeedExample] = []
    think_seed_train: list[SeedExample] = []
    think_seed_val: list[SeedExample] = []
    if args.chat_seed_dir:
        chat_seed_train, chat_seed_val = load_sft_seed_pool(
            Path(args.chat_seed_dir),
            max_rows_per_split=args.max_seed_rows_per_split,
        )
    if args.reasoning_seed_dir:
        think_seed_train, think_seed_val = load_sft_seed_pool(
            Path(args.reasoning_seed_dir),
            max_rows_per_split=args.max_seed_rows_per_split,
        )

    config_signature = {
        "chat_target_pairs": int(args.chat_target_pairs),
        "think_target_pairs": int(args.think_target_pairs),
        "tokenizer_path": file_signature(Path(args.tokenizer_path)),
        "chat_checkpoint": file_signature(Path(args.chat_checkpoint)) if args.chat_checkpoint else None,
        "think_checkpoint": file_signature(Path(args.think_checkpoint)) if args.think_checkpoint else None,
        "chat_seed_dir": {
            "train": file_signature(Path(args.chat_seed_dir) / "train.jsonl"),
            "val": file_signature(Path(args.chat_seed_dir) / "val.jsonl"),
        }
        if args.chat_seed_dir
        else None,
        "reasoning_seed_dir": {
            "train": file_signature(Path(args.reasoning_seed_dir) / "train.jsonl"),
            "val": file_signature(Path(args.reasoning_seed_dir) / "val.jsonl"),
        }
        if args.reasoning_seed_dir
        else None,
        "chat_max_new_tokens": args.chat_max_new_tokens,
        "think_max_new_tokens": args.think_max_new_tokens,
        "chat_temperature": args.chat_temperature,
        "think_temperature": args.think_temperature,
        "top_k": args.top_k,
        "max_prompt_chars": args.max_prompt_chars,
        "max_response_chars": args.max_response_chars,
        "min_alpha_ratio": args.min_alpha_ratio,
        "max_urls": args.max_urls,
        "max_code_fences": args.max_code_fences,
    }

    custom_meta_path = output_dir / "custom_negatives_meta.json"
    custom_train_path = output_dir / "custom_train.jsonl"
    custom_val_path = output_dir / "custom_val.jsonl"
    if (
        custom_meta_path.exists()
        and custom_train_path.exists()
        and custom_val_path.exists()
        and train_path.exists()
        and val_path.exists()
    ):
        existing_meta = json.loads(custom_meta_path.read_text(encoding="utf-8"))
        if existing_meta.get("config_signature") == config_signature:
            print(f"Custom negative mining already up to date at {output_dir}", flush=True)
            return

    tokenizer = None
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    run_stats: dict[str, Any] = {}

    base_val_examples = max(int(base_meta.get("val_examples", 0)), 1)
    base_train_examples = max(int(base_meta.get("train_examples", 0)), 1)
    val_ratio = base_val_examples / max(base_train_examples + base_val_examples, 1)

    def split_targets(total: int) -> tuple[int, int]:
        val_target = round(total * val_ratio)
        train_target = max(total - val_target, 0)
        return train_target, val_target

    if args.chat_target_pairs > 0:
        if not args.chat_checkpoint or not args.chat_seed_dir:
            raise ValueError("Chat custom negatives require both --chat-checkpoint and --chat-seed-dir.")
        if tokenizer is None:
            tokenizer = load_tokenizer(args.tokenizer_path)
        chat_model = load_model(Path(args.chat_checkpoint), device)
        chat_train_target, chat_val_target = split_targets(args.chat_target_pairs)
        mined_train, train_stats = mine_split(
            mode="chat",
            model=chat_model,
            tokenizer=tokenizer,
            device=device,
            seed_examples=chat_seed_train,
            target_count=chat_train_target,
            rng_seed=args.seed + 11,
            max_new_tokens=args.chat_max_new_tokens,
            temperature=args.chat_temperature,
            top_k=args.top_k,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            min_alpha_ratio=args.min_alpha_ratio,
            max_urls=args.max_urls,
            max_code_fences=args.max_code_fences,
        )
        mined_val, val_stats = mine_split(
            mode="chat",
            model=chat_model,
            tokenizer=tokenizer,
            device=device,
            seed_examples=chat_seed_val,
            target_count=chat_val_target,
            rng_seed=args.seed + 12,
            max_new_tokens=args.chat_max_new_tokens,
            temperature=args.chat_temperature,
            top_k=args.top_k,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            min_alpha_ratio=args.min_alpha_ratio,
            max_urls=args.max_urls,
            max_code_fences=args.max_code_fences,
        )
        train_rows.extend(mined_train)
        val_rows.extend(mined_val)
        run_stats["chat"] = {"train": train_stats, "val": val_stats}
        del chat_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.think_target_pairs > 0:
        if not args.think_checkpoint or not args.reasoning_seed_dir:
            raise ValueError("Think custom negatives require both --think-checkpoint and --reasoning-seed-dir.")
        if tokenizer is None:
            tokenizer = load_tokenizer(args.tokenizer_path)
        think_model = load_model(Path(args.think_checkpoint), device)
        think_train_target, think_val_target = split_targets(args.think_target_pairs)
        mined_train, train_stats = mine_split(
            mode="think",
            model=think_model,
            tokenizer=tokenizer,
            device=device,
            seed_examples=think_seed_train,
            target_count=think_train_target,
            rng_seed=args.seed + 21,
            max_new_tokens=args.think_max_new_tokens,
            temperature=args.think_temperature,
            top_k=args.top_k,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            min_alpha_ratio=args.min_alpha_ratio,
            max_urls=args.max_urls,
            max_code_fences=args.max_code_fences,
        )
        mined_val, val_stats = mine_split(
            mode="think",
            model=think_model,
            tokenizer=tokenizer,
            device=device,
            seed_examples=think_seed_val,
            target_count=think_val_target,
            rng_seed=args.seed + 22,
            max_new_tokens=args.think_max_new_tokens,
            temperature=args.think_temperature,
            top_k=args.top_k,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            min_alpha_ratio=args.min_alpha_ratio,
            max_urls=args.max_urls,
            max_code_fences=args.max_code_fences,
        )
        train_rows.extend(mined_train)
        val_rows.extend(mined_val)
        run_stats["think"] = {"train": train_stats, "val": val_stats}
        del think_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl(custom_train_path, train_rows)
    write_jsonl(custom_val_path, val_rows)
    concatenate_jsonl(train_path, [base_train_path, custom_train_path])
    concatenate_jsonl(val_path, [base_val_path, custom_val_path])

    combined_meta = json.loads(json.dumps(base_meta)) if base_meta else {}
    combined_meta["train_examples"] = int(base_meta.get("train_examples", 0)) + len(train_rows)
    combined_meta["val_examples"] = int(base_meta.get("val_examples", 0)) + len(val_rows)
    combined_meta["source_counts"] = summarize_source_counts(base_meta, train_rows, val_rows)
    combined_meta["custom_negative_mining"] = {
        "enabled": True,
        "config_signature": config_signature,
        "generated_train_examples": len(train_rows),
        "generated_val_examples": len(val_rows),
        "run_stats": run_stats,
    }
    (output_dir / "meta.json").write_text(json.dumps(combined_meta, indent=2) + "\n", encoding="utf-8")

    custom_meta = {
        "config_signature": config_signature,
        "generated_train_examples": len(train_rows),
        "generated_val_examples": len(val_rows),
        "run_stats": run_stats,
    }
    custom_meta_path.write_text(json.dumps(custom_meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(custom_meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
