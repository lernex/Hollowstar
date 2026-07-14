from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterator

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
MULTIBLANK_RE = re.compile(r"\n{3,}")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{18,}")


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
    allowed_languages: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    excluded_domains: tuple[str, ...] = ()
    max_turns: int | None = None
    skip_toxic: bool = False
    skip_redacted: bool = False
    prompt_column: str = "problem"
    response_column: str = "solution_wocode"
    s3_uri: str | None = None
    bucket: str | None = None
    target_examples: int | None = None
    max_examples: int | None = None
    filters: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        filters = raw.get("filters", {})
        return cls(
            name=raw["name"],
            dataset_name=raw["dataset_name"],
            split=raw["split"],
            weight=float(raw.get("weight", raw.get("target_examples", 1.0))),
            format=raw["format"],
            dataset_config=raw.get("dataset_config"),
            streaming=bool(raw.get("streaming", True)),
            keep_think=bool(raw.get("keep_think", False)),
            use_custom_instructions=bool(raw.get("use_custom_instructions", True)),
            allowed_languages=tuple(raw.get("allowed_languages", filters.get("allowed_languages", ()))),
            allowed_domains=tuple(raw.get("allowed_domains", filters.get("allowed_domains", ()))),
            excluded_domains=tuple(raw.get("excluded_domains", filters.get("excluded_domains", ()))),
            max_turns=int(raw["max_turns"]) if raw.get("max_turns") is not None else None,
            skip_toxic=bool(raw.get("skip_toxic", False)),
            skip_redacted=bool(raw.get("skip_redacted", False)),
            prompt_column=raw.get("prompt_column", raw.get("input_column", "problem")),
            response_column=raw.get("response_column", raw.get("output_column", "solution_wocode")),
            s3_uri=raw.get("s3_uri"),
            bucket=raw.get("bucket"),
            target_examples=raw.get("target_examples"),
            max_examples=raw.get("max_examples"),
            filters=filters,
        )


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTIBLANK_RE.sub("\n\n", text)
    return text.strip()


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


def language_allowed(language: Any, allowed_languages: tuple[str, ...]) -> bool:
    if not allowed_languages:
        return True
    if language is None:
        return False
    normalized = str(language).strip().lower()
    return normalized in {value.strip().lower() for value in allowed_languages}


def domain_allowed(row: dict[str, Any], spec: SourceSpec) -> bool:
    if not spec.allowed_domains and not spec.excluded_domains:
        return True
    candidates = []
    for key in ("domain", "domains", "category", "task_category", "subject", "source"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            candidates.extend(str(item).strip().lower() for item in value)
        else:
            candidates.append(str(value).strip().lower())
    allowed = {value.strip().lower() for value in spec.allowed_domains}
    excluded = {value.strip().lower() for value in spec.excluded_domains}
    if excluded and any(candidate in excluded for candidate in candidates):
        return False
    if allowed and not any(candidate in allowed for candidate in candidates):
        return False
    return True


def resolve_repo_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return Path(__file__).resolve().parents[1] / resolved


def download_s3_file(s3_uri: str, local_path: Path) -> None:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI for local SFT fallback, got {s3_uri!r}")
    bucket_and_key = s3_uri.removeprefix("s3://")
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI for local SFT fallback, got {s3_uri!r}")
    import boto3

    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(local_path))


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


def alpha_ratio(text: str) -> float:
    meaningful = [char for char in text if not char.isspace()]
    if not meaningful:
        return 0.0
    return sum(char.isalpha() for char in meaningful) / len(meaningful)


def latin_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 1.0
    latin_letters = sum(("a" <= char.lower() <= "z") for char in letters)
    return latin_letters / len(letters)


def looks_like_low_quality(text: str, *, min_alpha_ratio: float, max_urls: int, max_code_fences: int) -> bool:
    if not text.strip():
        return True
    if alpha_ratio(text) < min_alpha_ratio:
        return True
    if latin_letter_ratio(text) < 0.65:
        return True
    if REPEAT_CHAR_RE.search(text):
        return True
    if text.count("http://") + text.count("https://") + text.count("www.") > max_urls:
        return True
    if text.count("```") > max_code_fences:
        return True
    return False


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
            context_lines.append(f"{message['role'].capitalize()}: {content}")

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
    min_user_chars: int,
    min_assistant_chars: int,
    min_user_alpha_ratio: float,
    min_assistant_alpha_ratio: float,
    max_urls: int,
    max_code_fences: int,
) -> Iterator[dict[str, Any]]:
    normalized: list[dict[str, str]] = []
    for raw_message in messages:
        role = normalize_role(raw_message["role"])
        content = collapse_whitespace(raw_message["content"])
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
        if len(user_prompt) < min_user_chars or len(assistant) < min_assistant_chars:
            continue
        if looks_like_low_quality(
            user_prompt,
            min_alpha_ratio=min_user_alpha_ratio,
            max_urls=max_urls,
            max_code_fences=max_code_fences,
        ):
            continue
        if looks_like_low_quality(
            assistant,
            min_alpha_ratio=min_assistant_alpha_ratio,
            max_urls=max_urls,
            max_code_fences=max_code_fences,
        ):
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
    min_user_chars: int,
    min_assistant_chars: int,
    min_user_alpha_ratio: float,
    min_assistant_alpha_ratio: float,
    max_urls: int,
    max_code_fences: int,
) -> Iterator[dict[str, Any]]:
    if spec.format == "local_messages_jsonl":
        local_path = resolve_repo_path(spec.dataset_name)
        refresh_from_s3 = os.environ.get("METIS_REFRESH_LOCAL_SFT_S3", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if spec.s3_uri and (refresh_from_s3 or not local_path.exists()):
            print(f"[{spec.name}] Hydrating local SFT JSONL from {spec.s3_uri} into {local_path}.", flush=True)
            download_s3_file(spec.s3_uri, local_path)
        print(f"[{spec.name}] Loading local SFT JSONL from {local_path}.", flush=True)
        with local_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "messages" in row:
                    messages = row["messages"]
                else:
                    messages = [
                        {"role": "user", "content": row.get("user", "")},
                        {"role": "assistant", "content": row.get("assistant", "")},
                    ]
                try:
                    yield from iter_pairs_from_messages(
                        messages,
                        custom_instructions=str(row.get("custom_instructions", "")),
                        source_name=spec.name,
                        keep_think=spec.keep_think,
                        max_history_turns=max_history_turns,
                        max_user_chars=max_user_chars,
                        max_assistant_chars=max_assistant_chars,
                        max_think_chars=max_think_chars,
                        max_answer_chars=max_answer_chars,
                        min_user_chars=min_user_chars,
                        min_assistant_chars=min_assistant_chars,
                        min_user_alpha_ratio=min_user_alpha_ratio,
                        min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                        max_urls=max_urls,
                        max_code_fences=max_code_fences,
                    )
                except KeyError as error:
                    raise ValueError(f"Malformed local SFT row {line_number} in {local_path}: {error}") from error
        return

    force_local = os.environ.get("METIS_LOCAL_DATASETS", "").strip().lower() in {"1", "true", "yes", "on"}
    prefer_streaming = os.environ.get("METIS_PREFER_STREAMING_POSTTRAIN", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    resolved_streaming = spec.streaming if prefer_streaming else (False if force_local else spec.streaming)
    print(
        f"[{spec.name}] Loading post-train dataset via "
        f"{'streaming' if resolved_streaming else 'generic local eager'} path.",
        flush=True,
    )
    dataset = load_dataset(
        spec.dataset_name,
        name=spec.dataset_config,
        split=spec.split,
        streaming=resolved_streaming,
    )

    for row in dataset:
        if spec.allowed_languages and not language_allowed(row.get("language") or row.get("lang"), spec.allowed_languages):
            continue
        if not domain_allowed(row, spec):
            continue

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
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_user_alpha_ratio=min_user_alpha_ratio,
                min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                max_urls=max_urls,
                max_code_fences=max_code_fences,
            )
            continue

        if spec.format in {"messages", "tulu_messages"}:
            messages = row.get("messages") or row.get("conversation") or row.get("conversations") or []
            custom_instructions = row.get("system", "") if spec.use_custom_instructions else ""
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
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_user_alpha_ratio=min_user_alpha_ratio,
                min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                max_urls=max_urls,
                max_code_fences=max_code_fences,
            )
            continue

        if spec.format in {"prompt_response", "input_output", "sciriff_io"}:
            prompt = collapse_whitespace(str(row.get(spec.prompt_column, "")))
            response = collapse_whitespace(str(row.get(spec.response_column, "")))
            if not prompt or not response:
                continue
            yield from iter_pairs_from_messages(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                custom_instructions="",
                source_name=spec.name,
                keep_think=spec.keep_think,
                max_history_turns=max_history_turns,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_think_chars=max_think_chars,
                max_answer_chars=max_answer_chars,
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_user_alpha_ratio=min_user_alpha_ratio,
                min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                max_urls=max_urls,
                max_code_fences=max_code_fences,
            )
            continue

        if spec.format == "wildchat_conversation":
            if spec.skip_toxic and bool(row.get("toxic")):
                continue
            if spec.skip_redacted and bool(row.get("redacted")):
                continue

            raw_messages = row.get("conversation") or []
            if spec.allowed_languages:
                if any(
                    item.get("language") is not None
                    and not language_allowed(item.get("language"), spec.allowed_languages)
                    for item in raw_messages
                ):
                    continue

            conversation_turns = row.get("turn")
            if conversation_turns is None:
                conversation_turns = sum(
                    1
                    for item in raw_messages
                    if normalize_role(str(item.get("role", ""))) == "assistant"
                )
            if spec.max_turns is not None and int(conversation_turns) > spec.max_turns:
                continue

            messages = [
                {
                    "role": normalize_role(str(item.get("role", ""))),
                    "content": item.get("content", ""),
                }
                for item in raw_messages
            ]
            yield from iter_pairs_from_messages(
                messages,
                custom_instructions="",
                source_name=spec.name,
                keep_think=spec.keep_think,
                max_history_turns=max_history_turns,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_think_chars=max_think_chars,
                max_answer_chars=max_answer_chars,
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_user_alpha_ratio=min_user_alpha_ratio,
                min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                max_urls=max_urls,
                max_code_fences=max_code_fences,
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
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_user_alpha_ratio=min_user_alpha_ratio,
                min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                max_urls=max_urls,
                max_code_fences=max_code_fences,
            )
            continue

        if spec.format == "templategsm_solution":
            prompt = collapse_whitespace(str(row.get(spec.prompt_column, "")))
            response = collapse_whitespace(str(row.get(spec.response_column, "")))
            if not prompt or not response:
                continue
            yield from iter_pairs_from_messages(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                custom_instructions="",
                source_name=spec.name,
                keep_think=spec.keep_think,
                max_history_turns=max_history_turns,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_think_chars=max_think_chars,
                max_answer_chars=max_answer_chars,
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_user_alpha_ratio=min_user_alpha_ratio,
                min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                max_urls=max_urls,
                max_code_fences=max_code_fences,
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
                min_user_chars=min_user_chars,
                min_assistant_chars=min_assistant_chars,
                min_user_alpha_ratio=min_user_alpha_ratio,
                min_assistant_alpha_ratio=min_assistant_alpha_ratio,
                max_urls=max_urls,
                max_code_fences=max_code_fences,
            )
            continue

        raise ValueError(f"Unsupported Metis SFT format: {spec.format}")


def load_mixture(path: Path) -> tuple[int, int, list[SourceSpec]]:
    payload = json.loads(path.read_text())
    seed = int(payload.get("seed", 42))
    max_history_turns = int(payload.get("max_history_turns", 2))
    raw_sources = payload.get("sources")
    if raw_sources is None:
        raw_sources = []
        for bucket in payload.get("buckets") or []:
            for source in bucket.get("sources") or []:
                item = dict(source)
                item.setdefault("bucket", bucket.get("name"))
                item.setdefault("weight", source.get("target_examples", source.get("weight", 1.0)))
                raw_sources.append(item)
    sources = [SourceSpec.from_dict(item) for item in raw_sources]
    return seed, max_history_turns, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned Metis-1.3 chat or reasoning JSONL SFT data.")
    parser.add_argument("--mixture-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-examples", type=int, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-user-chars", type=int, default=1800)
    parser.add_argument("--max-assistant-chars", type=int, default=1200)
    parser.add_argument("--max-think-chars", type=int, default=520)
    parser.add_argument("--max-answer-chars", type=int, default=420)
    parser.add_argument("--min-user-chars", type=int, default=8)
    parser.add_argument("--min-assistant-chars", type=int, default=14)
    parser.add_argument("--min-user-alpha-ratio", type=float, default=0.42)
    parser.add_argument("--min-assistant-alpha-ratio", type=float, default=0.45)
    parser.add_argument("--max-urls", type=int, default=2)
    parser.add_argument("--max-code-fences", type=int, default=0)
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
                min_user_chars=args.min_user_chars,
                min_assistant_chars=args.min_assistant_chars,
                min_user_alpha_ratio=args.min_user_alpha_ratio,
                min_assistant_alpha_ratio=args.min_assistant_alpha_ratio,
                max_urls=args.max_urls,
                max_code_fences=args.max_code_fences,
            )
        )
        for source in sources
    ]
    source_buckets = [source.bucket or source.name for source in sources]
    planned_bucket_counts = Counter(source_buckets[index] for index in schedule)
    bucket_counts = Counter()

    def same_bucket_indices(source_index: int) -> list[int]:
        bucket = source_buckets[source_index]
        return [
            index
            for index, candidate_bucket in enumerate(source_buckets)
            if candidate_bucket == bucket
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
    dead_sources: set[int] = set()
    split_counts = Counter()
    source_counts = Counter()
    seen_hashes: set[str] = set()

    try:
        with train_tmp_path.open("w", encoding="utf-8") as train_handle, val_tmp_path.open("w", encoding="utf-8") as val_handle:
            def consume_source(source_index: int) -> bool:
                nonlocal accepted, duplicates
                iterator = iterators[source_index]
                try:
                    example = next(iterator)
                except StopIteration:
                    exhausted[sources[source_index].name] += 1
                    dead_sources.add(source_index)
                    return False

                signature = (
                    example["messages"][0]["content"].strip().lower()
                    + "\n<assistant>\n"
                    + example["messages"][1]["content"].strip().lower()
                )
                digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    duplicates += 1
                    return True
                seen_hashes.add(digest)

                payload = json.dumps(example["messages"], ensure_ascii=False, sort_keys=True)
                target_split = split_name(payload, args.val_ratio)
                handle = val_handle if target_split == "val" else train_handle
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")

                accepted += 1
                split_counts[target_split] += 1
                source_counts[example["source"]] += 1
                bucket_counts[source_buckets[source_index]] += 1

                if accepted == 1 or accepted % args.progress_interval == 0:
                    print(
                        f"Prepared Metis-1.3 SFT examples: {accepted}/{args.total_examples} | "
                        f"train={split_counts['train']} val={split_counts['val']} | "
                        f"latest_source={example['source']} duplicates={duplicates}",
                        flush=True,
                    )
                return True

            def consume_with_bucket_fallback(source_index: int) -> bool:
                for candidate_index in same_bucket_indices(source_index):
                    if candidate_index in dead_sources:
                        continue
                    if consume_source(candidate_index):
                        return True
                return False

            for source_index in schedule:
                if accepted >= args.total_examples:
                    break
                consume_with_bucket_fallback(source_index)

            if accepted < args.total_examples:
                for bucket, planned_count in planned_bucket_counts.items():
                    while accepted < args.total_examples and bucket_counts[bucket] < planned_count:
                        fallback_order = sorted(
                            [
                                index
                                for index, candidate_bucket in enumerate(source_buckets)
                                if candidate_bucket == bucket and index not in dead_sources
                            ],
                            key=lambda index: sources[index].weight,
                            reverse=True,
                        )
                        if not fallback_order:
                            break
                        advanced = False
                        accepted_before = accepted
                        for source_index in fallback_order:
                            if accepted >= args.total_examples or bucket_counts[bucket] >= planned_count:
                                break
                            advanced = consume_source(source_index) or advanced
                        if accepted == accepted_before:
                            break

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
        "planned_bucket_counts": dict(planned_bucket_counts),
        "bucket_counts": dict(bucket_counts),
        "max_history_turns": max_history_turns,
        "max_user_chars": args.max_user_chars,
        "max_assistant_chars": args.max_assistant_chars,
        "max_think_chars": args.max_think_chars,
        "max_answer_chars": args.max_answer_chars,
        "min_user_chars": args.min_user_chars,
        "min_assistant_chars": args.min_assistant_chars,
        "min_user_alpha_ratio": args.min_user_alpha_ratio,
        "min_assistant_alpha_ratio": args.min_assistant_alpha_ratio,
        "max_urls": args.max_urls,
        "max_code_fences": args.max_code_fences,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
