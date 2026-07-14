from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterator

from datasets import load_dataset


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
    data_dir: str | None = None
    streaming: bool = True
    allowed_languages: tuple[str, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    excluded_domains: tuple[str, ...] = ()
    drop_ties: bool = False
    drop_code: bool = False
    prefer_moderate_verbosity: bool = False
    max_turns: int | None = None
    pairing: str = "best_vs_worst"
    bucket: str | None = None
    target_pairs: int | None = None
    max_pairs: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        filters = raw.get("filters", {})
        return cls(
            name=raw["name"],
            dataset_name=raw["dataset_name"],
            split=raw.get("split", "train"),
            weight=float(raw.get("weight", raw.get("target_pairs", 1.0))),
            format=raw["format"],
            dataset_config=raw.get("dataset_config"),
            data_dir=raw.get("data_dir"),
            streaming=bool(raw.get("streaming", True)),
            allowed_languages=tuple(filters.get("allowed_languages", ())),
            allowed_domains=tuple(filters.get("allowed_domains", ())),
            excluded_domains=tuple(filters.get("excluded_domains", ())),
            drop_ties=bool(filters.get("drop_ties", False)),
            drop_code=bool(filters.get("drop_code", False)),
            prefer_moderate_verbosity=bool(filters.get("prefer_moderate_verbosity", False)),
            max_turns=int(filters["max_turns"]) if filters.get("max_turns") is not None else None,
            pairing=str(filters.get("pairing", "best_vs_worst")),
            bucket=raw.get("bucket"),
            target_pairs=raw.get("target_pairs"),
            max_pairs=raw.get("max_pairs"),
        )


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


def latin_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 1.0
    latin_letters = sum(("a" <= char.lower() <= "z") for char in letters)
    return latin_letters / len(letters)


def looks_like_low_quality(
    text: str,
    *,
    min_alpha_ratio: float,
    max_urls: int,
    max_code_fences: int,
) -> bool:
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


def language_allowed(language: Any, allowed_languages: tuple[str, ...]) -> bool:
    if not allowed_languages:
        return True
    if language is None:
        return False
    normalized = str(language).strip().lower()
    return normalized in {value.strip().lower() for value in allowed_languages}


def normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def build_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = normalize_role(str(message.get("role", "")))
        content = collapse_whitespace(str(message.get("content", "")))
        if role not in {"system", "user", "assistant"} or not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines).strip()


def message_content(message: dict[str, Any]) -> str:
    return collapse_whitespace(str(message.get("content") or message.get("value") or ""))


def prompt_and_response_from_messages(messages: Any) -> tuple[str, str]:
    if not isinstance(messages, list):
        return "", ""

    normalized: list[dict[str, str]] = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue
        role = normalize_role(str(raw_message.get("role") or raw_message.get("from") or ""))
        content = message_content(raw_message)
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})

    response = ""
    prompt_messages: list[dict[str, str]] = []
    for index in range(len(normalized) - 1, -1, -1):
        message = normalized[index]
        if message["role"] == "assistant":
            response = message["content"]
            prompt_messages = normalized[:index]
            break
    if not response:
        return build_prompt_from_messages(normalized), ""
    return build_prompt_from_messages(prompt_messages), response


def maybe_strip_assistant_prefix(prompt: str) -> str:
    prompt = prompt.strip()
    if prompt.endswith("Assistant:"):
        return prompt[: -len("Assistant:")].rstrip()
    return prompt


def normalize_nectar_prompt(prompt: str) -> str:
    cooked = prompt.replace("\n\nHuman:", "\nUser:").replace("\n\nAssistant:", "\nAssistant:")
    cooked = cooked.replace("Human:", "User:")
    return maybe_strip_assistant_prefix(collapse_whitespace(cooked))


def preference_payload(
    *,
    prompt: str,
    chosen: str,
    rejected: str,
    source_name: str,
    pair_weight: float,
) -> dict[str, Any] | None:
    prompt = collapse_whitespace(prompt)
    chosen = collapse_whitespace(chosen)
    rejected = collapse_whitespace(rejected)
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": source_name,
        "pair_weight": float(pair_weight),
    }


def load_source_dataset(spec: SourceSpec):
    force_local = os.environ.get("METIS_LOCAL_DATASETS", "").strip().lower() in {"1", "true", "yes", "on"}
    prefer_streaming = os.environ.get("METIS_PREFER_STREAMING_POSTTRAIN", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    resolved_streaming = spec.streaming if prefer_streaming else (False if force_local else spec.streaming)
    print(
        f"[{spec.name}] Loading preference dataset via "
        f"{'streaming' if resolved_streaming else 'generic local eager'} path.",
        flush=True,
    )
    if spec.format == "helpsteer2_preference" and spec.data_dir is None:
        try:
            return load_dataset(spec.dataset_name, split=spec.split, streaming=resolved_streaming, data_dir="preference")
        except Exception:
            pass
    return load_dataset(
        spec.dataset_name,
        name=spec.dataset_config,
        split=spec.split,
        streaming=resolved_streaming,
        data_dir=spec.data_dir,
    )


def extract_numeric_rating(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    if isinstance(value, list):
        if not value:
            return 0.0
        return max(extract_numeric_rating(item) for item in value)
    if isinstance(value, dict):
        if "Rating" in value:
            return extract_numeric_rating(value["Rating"])
        if "score" in value:
            return extract_numeric_rating(value["score"])
    return 0.0


def contains_code_like_content(*texts: str) -> bool:
    needles = ("```", "def ", "class ", "import ", "#include", "SELECT ", "function ", "public ")
    cooked = "\n".join(texts)
    return any(needle in cooked for needle in needles)


def iter_helpsteer3_pairs(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    dataset = load_source_dataset(spec)
    for row in dataset:
        if not language_allowed(row.get("language"), spec.allowed_languages):
            continue
        domain = str(row.get("domain", "")).strip().lower()
        allowed_domains = {value.strip().lower() for value in spec.allowed_domains}
        excluded_domains = {value.strip().lower() for value in spec.excluded_domains}
        if spec.allowed_domains and domain not in allowed_domains:
            continue
        if spec.excluded_domains and domain in excluded_domains:
            continue
        prompt = build_prompt_from_messages(list(row.get("context") or []))
        response1 = str(row.get("response1", ""))
        response2 = str(row.get("response2", ""))
        if spec.drop_code and contains_code_like_content(prompt, response1, response2):
            continue
        preference = int(row.get("overall_preference", 0))
        if preference == 0 and spec.drop_ties:
            continue
        if preference < 0:
            chosen, rejected = response1, response2
        elif preference > 0:
            chosen, rejected = response2, response1
        else:
            continue
        payload = preference_payload(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source_name=spec.name,
            pair_weight=max(abs(preference), 1),
        )
        if payload is not None:
            yield payload


def iter_helpsteer2_pairs(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    dataset = load_source_dataset(spec)
    if spec.data_dir == "preference":
        for row in dataset:
            prompt = collapse_whitespace(str(row.get("prompt", "")))
            response1 = str(row.get("response_1") or row.get("chosen_response") or row.get("chosen") or "")
            response2 = str(row.get("response_2") or row.get("rejected_response") or row.get("rejected") or "")
            preference_strength = extract_numeric_rating(row.get("preference_strength"))
            if preference_strength > 0:
                chosen, rejected = response2, response1
            elif preference_strength < 0:
                chosen, rejected = response1, response2
            else:
                continue
            if spec.drop_code and contains_code_like_content(prompt, chosen, rejected):
                continue
            payload = preference_payload(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                source_name=spec.name,
                pair_weight=max(abs(preference_strength), 1.0),
            )
            if payload is not None:
                yield payload
        return

    current_prompt: str | None = None
    bucket: list[tuple[float, str]] = []

    def flush_bucket() -> Iterator[dict[str, Any]]:
        if current_prompt is None or len(bucket) < 2:
            return
        ranked = sorted(bucket, key=lambda item: item[0], reverse=True)
        best_score, best_response = ranked[0]
        worst_score, worst_response = ranked[-1]
        if best_score <= worst_score:
            return
        payload = preference_payload(
            prompt=current_prompt,
            chosen=best_response,
            rejected=worst_response,
            source_name=spec.name,
            pair_weight=max(best_score - worst_score, 1.0),
        )
        if payload is not None:
            yield payload

    for row in dataset:
        prompt = collapse_whitespace(str(row.get("prompt", "")))
        response = str(row.get("response", ""))
        if not prompt or not response:
            continue
        if spec.drop_code and contains_code_like_content(prompt, response):
            continue
        score = (
            extract_numeric_rating(row.get("helpfulness"))
            + extract_numeric_rating(row.get("correctness"))
            + extract_numeric_rating(row.get("coherence"))
        )
        if current_prompt is None:
            current_prompt = prompt
        if prompt != current_prompt:
            yield from flush_bucket()
            current_prompt = prompt
            bucket = []
        bucket.append((score, response))
    yield from flush_bucket()


def extract_ultrafeedback_completion_score(completion: dict[str, Any], *, prefer_moderate_verbosity: bool) -> float:
    annotations = completion.get("annotations") or {}
    total = 0.0
    for key in ["instruction_following", "truthfulness", "honesty", "helpfulness"]:
        total += extract_numeric_rating(annotations.get(key))
    if prefer_moderate_verbosity:
        response = str(completion.get("response", ""))
        if len(response) > 1600:
            total -= 1.0
        elif len(response) < 80:
            total -= 0.5
    return total


def iter_ultrafeedback_pairs(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    dataset = load_source_dataset(spec)
    for row in dataset:
        prompt = collapse_whitespace(str(row.get("instruction", "")))
        completions = list(row.get("completions") or [])
        if not prompt or len(completions) < 2:
            continue
        candidates: list[tuple[float, str]] = []
        for completion in completions:
            response = str(completion.get("response", ""))
            if not response:
                continue
            if spec.drop_code and contains_code_like_content(prompt, response):
                continue
            score = extract_ultrafeedback_completion_score(
                completion,
                prefer_moderate_verbosity=spec.prefer_moderate_verbosity,
            )
            candidates.append((score, response))
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        best_score, chosen = ranked[0]
        worst_score, rejected = ranked[-1]
        if best_score <= worst_score:
            continue
        payload = preference_payload(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source_name=spec.name,
            pair_weight=max(best_score - worst_score, 1.0),
        )
        if payload is not None:
            yield payload


def iter_nectar_pairs(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    dataset = load_source_dataset(spec)
    rng = Random(13)
    for row in dataset:
        if row.get("good_natured") is False:
            continue
        turns = int(row.get("turns", 1))
        if spec.max_turns is not None and turns > spec.max_turns:
            continue
        prompt = normalize_nectar_prompt(str(row.get("prompt", "")))
        answers = list(row.get("answers") or [])
        if not prompt or len(answers) < 2:
            continue
        candidates = []
        for answer in answers:
            response = str(answer.get("answer", ""))
            rank = int(answer.get("rank", 999))
            if spec.drop_code and contains_code_like_content(prompt, response):
                continue
            candidates.append((rank, response))
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda item: item[0])
        best_rank, chosen = ranked[0]
        rejected_items: list[tuple[int, str]] = []
        if spec.pairing == "top_vs_bottom_or_top_vs_sampled_negative" and len(ranked) > 2:
            negative_pool = ranked[max(1, len(ranked) // 2) :]
            rejected_items.append(ranked[-1])
            sampled = rng.choice(negative_pool)
            if sampled[1] != ranked[-1][1]:
                rejected_items.append(sampled)
        else:
            rejected_items.append(ranked[-1])
        for worst_rank, rejected in rejected_items:
            if best_rank >= worst_rank:
                continue
            payload = preference_payload(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                source_name=spec.name,
                pair_weight=max(worst_rank - best_rank, 1.0),
            )
            if payload is not None:
                yield payload


def iter_openr1_math_verify_pairs(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    dataset = load_source_dataset(spec)
    for row in dataset:
        prompt = collapse_whitespace(str(row.get("problem") or row.get("prompt") or ""))
        generations = list(row.get("generations") or [])
        correctness = list(row.get("correctness_math_verify") or [])
        if not prompt or len(generations) < 2 or len(correctness) != len(generations):
            continue
        chosen = ""
        rejected = ""
        for generation, is_correct in zip(generations, correctness):
            response = str(generation or "")
            if not response:
                continue
            if bool(is_correct) and not chosen:
                chosen = response
            if not bool(is_correct) and not rejected:
                rejected = response
            if chosen and rejected:
                break
        if spec.drop_code and contains_code_like_content(prompt, chosen, rejected):
            continue
        payload = preference_payload(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source_name=spec.name,
            pair_weight=2.0,
        )
        if payload is not None:
            yield payload


def iter_prompt_chosen_rejected_pairs(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    dataset = load_source_dataset(spec)
    for row in dataset:
        if not language_allowed(row.get("language") or row.get("lang"), spec.allowed_languages):
            continue
        prompt = (
            str(row.get("prompt") or row.get("instruction") or row.get("question") or row.get("input") or "")
        )
        chosen_raw = row.get("chosen") or row.get("chosen_response") or row.get("response_chosen") or ""
        rejected_raw = row.get("rejected") or row.get("rejected_response") or row.get("response_rejected") or ""
        chosen_prompt = ""
        rejected_prompt = ""
        if isinstance(chosen_raw, list):
            chosen_prompt, chosen = prompt_and_response_from_messages(chosen_raw)
        elif isinstance(chosen_raw, dict):
            chosen = message_content(chosen_raw)
        else:
            chosen = str(chosen_raw)
        if isinstance(rejected_raw, list):
            rejected_prompt, rejected = prompt_and_response_from_messages(rejected_raw)
        elif isinstance(rejected_raw, dict):
            rejected = message_content(rejected_raw)
        else:
            rejected = str(rejected_raw)
        if not prompt:
            prompt = chosen_prompt or rejected_prompt
        if spec.drop_code and contains_code_like_content(prompt, chosen, rejected):
            continue
        payload = preference_payload(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            source_name=spec.name,
            pair_weight=1.0,
        )
        if payload is not None:
            yield payload


def iter_source_pairs(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    if spec.format == "helpsteer3_preference":
        yield from iter_helpsteer3_pairs(spec)
        return
    if spec.format == "helpsteer2_preference":
        yield from iter_helpsteer2_pairs(spec)
        return
    if spec.format == "ultrafeedback_pairwise":
        yield from iter_ultrafeedback_pairs(spec)
        return
    if spec.format == "nectar_ranked":
        yield from iter_nectar_pairs(spec)
        return
    if spec.format == "openr1_math_verify_preference":
        yield from iter_openr1_math_verify_pairs(spec)
        return
    if spec.format in {"prompt_chosen_rejected", "skywork_preference", "cleaned_preference"}:
        yield from iter_prompt_chosen_rejected_pairs(spec)
        return
    raise ValueError(f"Unsupported Metis preference format: {spec.format}")


def load_mixture(path: Path) -> tuple[int, list[SourceSpec]]:
    payload = json.loads(path.read_text())
    seed = int(payload.get("seed", 42))
    raw_sources = payload.get("default_sources")
    if raw_sources is None:
        raw_sources = []
        for bucket in payload.get("buckets") or []:
            for source in bucket.get("sources") or []:
                item = dict(source)
                item.setdefault("bucket", bucket.get("name"))
                item.setdefault("weight", source.get("target_pairs", source.get("weight", 1.0)))
                raw_sources.append(item)
    sources = [SourceSpec.from_dict(item) for item in raw_sources]
    return seed, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cleaned Metis-1.5 preference JSONL for reward-model and DPO.")
    parser.add_argument("--mixture-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-pairs", type=int, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-prompt-chars", type=int, default=2200)
    parser.add_argument("--max-response-chars", type=int, default=1600)
    parser.add_argument("--min-prompt-chars", type=int, default=8)
    parser.add_argument("--min-response-chars", type=int, default=14)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.42)
    parser.add_argument("--max-urls", type=int, default=2)
    parser.add_argument("--max-code-fences", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=2000)
    args = parser.parse_args()

    seed, sources = load_mixture(Path(args.mixture_config))
    schedule = build_schedule([source.weight for source in sources], args.total_pairs, seed)
    iterators = [iter(iter_source_pairs(source)) for source in sources]
    source_buckets = [source.bucket or source.name for source in sources]
    planned_bucket_counter: dict[str, int] = {}
    for source_index in schedule:
        bucket = source_buckets[source_index]
        planned_bucket_counter[bucket] = planned_bucket_counter.get(bucket, 0) + 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_handle = (output_dir / "train.jsonl").open("w", encoding="utf-8")
    val_handle = (output_dir / "val.jsonl").open("w", encoding="utf-8")

    train_count = 0
    val_count = 0
    accepted = 0
    exhausted = {source.name: 0 for source in sources}
    dead_sources: set[int] = set()
    source_counts = {source.name: {"train": 0, "val": 0} for source in sources}
    bucket_counts = {bucket: 0 for bucket in source_buckets}

    def same_bucket_indices(source_index: int) -> list[int]:
        bucket = source_buckets[source_index]
        return [
            index
            for index, candidate_bucket in enumerate(source_buckets)
            if candidate_bucket == bucket
        ]

    try:
        def consume_source(source_index: int, attempted: int) -> bool:
            nonlocal train_count, val_count, accepted
            try:
                pair = next(iterators[source_index])
            except StopIteration:
                exhausted[sources[source_index].name] += 1
                dead_sources.add(source_index)
                return False

            prompt = collapse_whitespace(pair["prompt"])[: args.max_prompt_chars]
            chosen = collapse_whitespace(pair["chosen"])[: args.max_response_chars]
            rejected = collapse_whitespace(pair["rejected"])[: args.max_response_chars]

            if len(prompt) < args.min_prompt_chars:
                return True
            if len(chosen) < args.min_response_chars or len(rejected) < args.min_response_chars:
                return True
            if looks_like_low_quality(
                prompt,
                min_alpha_ratio=args.min_alpha_ratio,
                max_urls=args.max_urls,
                max_code_fences=args.max_code_fences,
            ):
                return True
            if looks_like_low_quality(
                chosen,
                min_alpha_ratio=args.min_alpha_ratio,
                max_urls=args.max_urls,
                max_code_fences=args.max_code_fences,
            ):
                return True
            if looks_like_low_quality(
                rejected,
                min_alpha_ratio=args.min_alpha_ratio,
                max_urls=args.max_urls,
                max_code_fences=args.max_code_fences,
            ):
                return True

            payload = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "pair_weight": float(pair.get("pair_weight", 1.0)),
                "source": pair["source"],
            }
            split = split_name(f"{prompt}\n\n{chosen}\n\n{rejected}", args.val_ratio)
            target = train_handle if split == "train" else val_handle
            target.write(json.dumps(payload, ensure_ascii=False) + "\n")
            source_counts[pair["source"]][split] += 1
            bucket_counts[source_buckets[source_index]] = bucket_counts.get(source_buckets[source_index], 0) + 1
            if split == "train":
                train_count += 1
            else:
                val_count += 1
            accepted += 1

            if accepted == 1 or accepted % args.progress_interval == 0:
                print(
                    f"Prepared preference pairs {accepted}/{args.total_pairs} | "
                    f"train={train_count} val={val_count}",
                    flush=True,
                )
            return True

        def consume_with_bucket_fallback(source_index: int, attempted: int) -> bool:
            for candidate_index in same_bucket_indices(source_index):
                if candidate_index in dead_sources:
                    continue
                if consume_source(candidate_index, attempted):
                    return True
            return False

        for index, source_index in enumerate(schedule, start=1):
            if accepted >= args.total_pairs:
                break
            consume_with_bucket_fallback(source_index, index)

        if accepted < args.total_pairs:
            attempted = len(schedule)
            for bucket, planned_count in planned_bucket_counter.items():
                while accepted < args.total_pairs and bucket_counts.get(bucket, 0) < planned_count:
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
                    accepted_before = accepted
                    for source_index in fallback_order:
                        if accepted >= args.total_pairs or bucket_counts.get(bucket, 0) >= planned_count:
                            break
                        attempted += 1
                        consume_source(source_index, attempted)
                    if accepted == accepted_before:
                        break
    finally:
        train_handle.close()
        val_handle.close()

    meta = {
        "mixture_config": args.mixture_config,
        "total_pairs_requested": args.total_pairs,
        "train_examples": train_count,
        "val_examples": val_count,
        "val_ratio": args.val_ratio,
        "exhausted_sources": exhausted,
        "source_counts": source_counts,
        "planned_bucket_counts": planned_bucket_counter,
        "bucket_counts": bucket_counts,
        "max_prompt_chars": args.max_prompt_chars,
        "max_response_chars": args.max_response_chars,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
