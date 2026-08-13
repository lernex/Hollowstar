from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Iterable, Iterator

from .dedup import canonical_text
from .code_dedup import code_tokens


WORD_RE = re.compile(r"\w+", re.UNICODE)
CODE_MARKER_RE = re.compile(
    r"(?m)(?:^\s*(?:def|class|function|fn|public|private|package|import|from|#include)\b|"
    r"(?:assert|return|throw|raise)\s+|=>|::|\{[^}]*\}|\[[^]]*\]\s*=)"
)
CODE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
CODE_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
CODE_STRING_RE = re.compile(r'''^(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)$''')
CODE_KEYWORDS = frozenset(
    "and as assert async await break case catch class const continue def delete do else elif enum "
    "except export extends false finally fn for from function go if import in interface let match "
    "namespace new nil none not null of or package private protected public raise return static "
    "struct super switch this throw trait true try type typeof use var while with yield".split()
)
HOLDOUT_GROUP_DIGEST_BYTES = 16

GENEALOGY_STRICT_KEYS = frozenset(
    {
        "benchmark",
        "benchmark_id",
        "benchmark_name",
        "eval_benchmark",
        "eval_dataset",
        "evaluation_benchmark",
        "evaluation_dataset",
        "seed_benchmark",
        "source_benchmark",
    }
)
GENEALOGY_DATASET_KEYS = frozenset(
    {
        "dataset",
        "dataset_id",
        "dataset_name",
        "datasets",
        "origin_dataset",
        "seed_dataset",
        "source_dataset",
        "task_dataset",
        "upstream_dataset",
    }
)
# These names are real benchmark IDs but also ordinary domain/task words.  We
# reject them only when upstream metadata explicitly labels the field as a
# benchmark/evaluation source, never from a generic `dataset` field.
AMBIGUOUS_BENCHMARK_ALIASES = frozenset(
    {"apps", "arc", "drop", "frames", "math", "race"}
)
_ALIAS_SEPARATORS = re.compile(r"[^a-z0-9]+")


def _genealogy_key(value: Any) -> str:
    return _ALIAS_SEPARATORS.sub("_", str(value or "").strip().lower()).strip("_")


def benchmark_genealogy_aliases(
    registry: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """Build conservative aliases for explicit upstream benchmark lineage.

    The first mapping is safe in dataset-lineage fields.  The second also
    contains short ambiguous benchmark IDs and is used only for fields whose
    name explicitly says benchmark/evaluation.
    """

    safe: dict[str, str] = {}
    strict: dict[str, str] = {}
    for benchmark in registry.get("benchmarks", []):
        if not isinstance(benchmark, Mapping):
            continue
        benchmark_id = str(benchmark.get("id") or "").strip()
        if not benchmark_id:
            continue
        candidates = {benchmark_id}
        repo_id = str(benchmark.get("repo_id") or "").strip()
        if repo_id:
            candidates.add(repo_id)
            candidates.add(repo_id.rsplit("/", 1)[-1])
        for config in benchmark.get("configs", []):
            config_key = _genealogy_key(config)
            benchmark_key = _genealogy_key(benchmark_id)
            if benchmark_key and benchmark_key in config_key:
                candidates.add(str(config))
        for job in benchmark.get("jobs", []):
            if isinstance(job, Mapping):
                config = str(job.get("config") or "")
                config_key = _genealogy_key(config)
                if _genealogy_key(benchmark_id) in config_key:
                    candidates.add(config)
        for candidate in candidates:
            alias = _genealogy_key(candidate)
            if not alias:
                continue
            strict[alias] = benchmark_id
            if alias not in AMBIGUOUS_BENCHMARK_ALIASES:
                safe[alias] = benchmark_id
    return safe, strict


def _genealogy_values(
    value: Any,
    *,
    depth: int = 0,
) -> Iterator[tuple[bool, str]]:
    if depth > 6:
        return
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = _genealogy_key(raw_key)
            if key in GENEALOGY_STRICT_KEYS:
                for scalar in _scalar_lineage_values(nested, depth=depth + 1):
                    yield True, scalar
            elif key in GENEALOGY_DATASET_KEYS:
                for scalar in _scalar_lineage_values(nested, depth=depth + 1):
                    yield False, scalar
            if isinstance(nested, (Mapping, list, tuple)):
                yield from _genealogy_values(nested, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for nested in value[:250]:
            yield from _genealogy_values(nested, depth=depth + 1)


def _scalar_lineage_values(value: Any, *, depth: int) -> Iterator[str]:
    if depth > 6:
        return
    if isinstance(value, str):
        if 0 < len(value) <= 2048:
            yield value
    elif isinstance(value, Mapping):
        for nested in list(value.values())[:250]:
            yield from _scalar_lineage_values(nested, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for nested in value[:250]:
            yield from _scalar_lineage_values(nested, depth=depth + 1)


def benchmark_genealogy_match(
    metadata: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> str | None:
    """Return the pinned benchmark ID named by explicit lineage metadata."""

    safe, strict = benchmark_genealogy_aliases(registry)
    for strict_field, value in _genealogy_values(metadata):
        normalized = _genealogy_key(value)
        aliases = strict if strict_field else safe
        # Match the full value and URL/repository-like suffixes.  This catches
        # `openai/gsm8k`, HF URLs, and lists without substring-matching prose.
        pieces = [normalized]
        raw_parts = [part for part in re.split(r"[/\\|,;\s]+", value) if part]
        pieces.extend(_genealogy_key(part.split("?", 1)[0]) for part in raw_parts)
        if len(raw_parts) >= 2:
            pieces.append(_genealogy_key("/".join(raw_parts[-2:])))
        for piece in pieces:
            if piece in aliases:
                return aliases[piece]
    return None


def looks_like_code(text: str) -> bool:
    return bool(CODE_MARKER_RE.search(text)) or sum(text.count(character) for character in "{}();=") >= 4


def ngram_hashes(text: str, size: int = 13) -> set[int]:
    words = WORD_RE.findall(canonical_text(text))
    if len(words) < size:
        return set()
    return {
        int.from_bytes(hashlib.blake2b(" ".join(words[index : index + size]).encode(), digest_size=8).digest(), "little")
        for index in range(len(words) - size + 1)
    }


def code_ngram_hashes(text: str, size: int = 12) -> set[int]:
    tokens = code_tokens(text)
    if len(tokens) < size:
        return set()
    return {
        int.from_bytes(
            hashlib.blake2b("\0".join(tokens[index : index + size]).encode(), digest_size=8).digest(),
            "little",
        )
        for index in range(len(tokens) - size + 1)
    }


def code_skeleton_ngram_hashes(text: str, size: int = 16) -> set[int]:
    """Hash code structure while ignoring renamed identifiers and literals.

    This catches benchmark solutions copied with variable renaming or changed
    constants without introducing embedding/semantic matching.
    """

    skeleton: list[str] = []
    for token in code_tokens(text):
        lowered = token.lower()
        if CODE_IDENTIFIER_RE.fullmatch(token):
            skeleton.append(lowered if lowered in CODE_KEYWORDS else "<id>")
        elif CODE_NUMBER_RE.fullmatch(token):
            skeleton.append("<num>")
        elif CODE_STRING_RE.fullmatch(token):
            skeleton.append("<str>")
        else:
            skeleton.append(token)
    if len(skeleton) < size:
        return set()
    return {
        int.from_bytes(
            hashlib.blake2b("\0".join(skeleton[index : index + size]).encode(), digest_size=8).digest(),
            "little",
        )
        for index in range(len(skeleton) - size + 1)
    }


def _holdout_group_and_text(
    holdout: str | Mapping[str, Any] | tuple[str, str],
    index: int,
) -> tuple[bytes, str]:
    """Return a stable benchmark-row identity and one indexed text fragment.

    Production holdout records carry ``metadata.holdout_row_id`` so prompt,
    context, answer, and code fragments from the same evaluation example may
    contribute to the same threshold. Plain strings remain supported for small
    tests and are deliberately treated as distinct rows.
    """

    if isinstance(holdout, str):
        group_value = f"fragment:{index}"
        text = holdout
    elif (
        isinstance(holdout, tuple)
        and len(holdout) == 2
        and isinstance(holdout[0], str)
        and isinstance(holdout[1], str)
    ):
        group_value, text = holdout
    elif isinstance(holdout, Mapping):
        text = str(holdout.get("text") or "")
        metadata = holdout.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        group_value = str(
            metadata.get("holdout_row_id")
            or holdout.get("holdout_row_id")
            or holdout.get("id")
            or f"fragment:{index}"
        )
    else:
        raise TypeError(
            "Holdouts must be strings, (row_id, text) pairs, or mappings with text"
        )
    group_digest = hashlib.sha256(group_value.encode("utf-8")).digest()[
        :HOLDOUT_GROUP_DIGEST_BYTES
    ]
    return group_digest, text


def _freeze_postings(
    postings: Mapping[int, set[bytes]],
    *,
    maximum_shingle_rows: int,
) -> tuple[dict[int, frozenset[bytes]], int]:
    """Drop corpus-generic shingles before freezing row-aware postings."""

    retained: dict[int, frozenset[bytes]] = {}
    suppressed = 0
    for shingle, groups in postings.items():
        if len(groups) > maximum_shingle_rows:
            suppressed += 1
        else:
            retained[int(shingle)] = frozenset(groups)
    return retained, suppressed


def required_matches(minimum: int, total_ngrams: int, fraction: float = 0.0) -> int:
    """Scale the match threshold with the document, not against it.

    A fixed count is the wrong shape for this test. A document of n n-grams gets
    n chances to collide with the evaluation set, so a constant threshold makes
    a 400 KB source file far likelier to trip than a 3 KB one -- which is
    exactly the bias measured on the 1.6 corpus, where decontamination kept
    49.9% of documents but only 21.4% of characters. Current practice is
    proportional: discard when a share of the document's own n-grams matches,
    rather than when some absolute number does.

    The floor still applies, so short documents cannot be cleared by arithmetic.
    fraction=0.0 reproduces the old absolute behaviour exactly.
    """

    if fraction <= 0.0:
        return minimum
    return max(minimum, math.ceil(fraction * total_ngrams))


def _matches_one_holdout_group(
    postings: Mapping[int, frozenset[bytes]],
    candidates: set[int],
    minimum: int,
    fraction: float = 0.0,
) -> bool:
    """Require all threshold matches to belong to one evaluation row."""

    threshold = required_matches(minimum, len(candidates), fraction)
    group_counts: dict[bytes, int] = {}
    for candidate in candidates:
        for group in postings.get(candidate, ()):
            count = group_counts.get(group, 0) + 1
            if count >= threshold:
                return True
            group_counts[group] = count
    return False


@dataclass(frozen=True)
class ContaminationIndex:
    exact: frozenset[str]
    ngram_postings: Mapping[int, frozenset[bytes]]
    ngram_size: int = 13
    minimum_matching_ngrams: int = 2
    short_ngram_postings: Mapping[int, frozenset[bytes]] = field(default_factory=dict)
    short_ngram_size: int = 8
    minimum_short_matching_ngrams: int = 2
    code_ngram_postings: Mapping[int, frozenset[bytes]] = field(default_factory=dict)
    code_ngram_size: int = 12
    minimum_code_matching_ngrams: int = 2
    code_skeleton_ngram_postings: Mapping[int, frozenset[bytes]] = field(default_factory=dict)
    code_skeleton_ngram_size: int = 16
    minimum_code_skeleton_matching_ngrams: int = 2
    maximum_shingle_rows: int = 32
    # Detection tuning, deliberately not part of the holdout bundle's identity.
    # What is withheld from training is release-immutable; how overlap is
    # detected is tuning, and binding the two together meant retuning a
    # threshold cost a full corpus rebuild. See docs 0a.
    match_fraction: float = 0.0
    suppressed_shingles: Mapping[str, int] = field(default_factory=dict)

    @property
    def ngrams(self) -> frozenset[int]:
        return frozenset(self.ngram_postings)

    @property
    def short_ngrams(self) -> frozenset[int]:
        return frozenset(self.short_ngram_postings)

    @property
    def code_ngrams(self) -> frozenset[int]:
        return frozenset(self.code_ngram_postings)

    @property
    def code_skeleton_ngrams(self) -> frozenset[int]:
        return frozenset(self.code_skeleton_ngram_postings)

    @classmethod
    def build(
        cls,
        holdouts: Iterable[str | Mapping[str, Any] | tuple[str, str]],
        *,
        ngram_size: int = 13,
        minimum_matching_ngrams: int = 2,
        short_ngram_size: int = 8,
        minimum_short_matching_ngrams: int = 2,
        code_ngram_size: int = 12,
        minimum_code_matching_ngrams: int = 2,
        code_skeleton_ngram_size: int = 16,
        minimum_code_skeleton_matching_ngrams: int = 2,
        maximum_shingle_rows: int = 32,
        match_fraction: float = 0.0,
    ) -> "ContaminationIndex":
        if maximum_shingle_rows < 1:
            raise ValueError("maximum_shingle_rows must be positive")
        for name, value in (
            ("minimum_matching_ngrams", minimum_matching_ngrams),
            ("minimum_short_matching_ngrams", minimum_short_matching_ngrams),
            ("minimum_code_matching_ngrams", minimum_code_matching_ngrams),
            (
                "minimum_code_skeleton_matching_ngrams",
                minimum_code_skeleton_matching_ngrams,
            ),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        exact: set[str] = set()
        ngrams: dict[int, set[bytes]] = defaultdict(set)
        short_ngrams: dict[int, set[bytes]] = defaultdict(set)
        code_ngrams: dict[int, set[bytes]] = defaultdict(set)
        code_skeleton_ngrams: dict[int, set[bytes]] = defaultdict(set)
        for holdout_index, holdout in enumerate(holdouts):
            group, text = _holdout_group_and_text(holdout, holdout_index)
            normalized = canonical_text(text)
            if not normalized:
                continue
            exact.add(hashlib.sha256(normalized.encode()).hexdigest())
            for shingle in ngram_hashes(normalized, ngram_size):
                ngrams[shingle].add(group)
            for shingle in ngram_hashes(normalized, short_ngram_size):
                short_ngrams[shingle].add(group)
            if looks_like_code(text):
                for shingle in code_ngram_hashes(text, code_ngram_size):
                    code_ngrams[shingle].add(group)
                for shingle in code_skeleton_ngram_hashes(
                    text, code_skeleton_ngram_size
                ):
                    code_skeleton_ngrams[shingle].add(group)
        frozen_ngrams, suppressed_ngrams = _freeze_postings(
            ngrams, maximum_shingle_rows=maximum_shingle_rows
        )
        frozen_short, suppressed_short = _freeze_postings(
            short_ngrams, maximum_shingle_rows=maximum_shingle_rows
        )
        frozen_code, suppressed_code = _freeze_postings(
            code_ngrams, maximum_shingle_rows=maximum_shingle_rows
        )
        frozen_code_skeleton, suppressed_code_skeleton = _freeze_postings(
            code_skeleton_ngrams, maximum_shingle_rows=maximum_shingle_rows
        )
        return cls(
            frozenset(exact),
            frozen_ngrams,
            ngram_size,
            minimum_matching_ngrams,
            frozen_short,
            short_ngram_size,
            minimum_short_matching_ngrams,
            frozen_code,
            code_ngram_size,
            minimum_code_matching_ngrams,
            frozen_code_skeleton,
            code_skeleton_ngram_size,
            minimum_code_skeleton_matching_ngrams,
            maximum_shingle_rows,
            match_fraction,
            {
                "ngrams": suppressed_ngrams,
                "short_ngrams": suppressed_short,
                "code_ngrams": suppressed_code,
                "code_skeleton_ngrams": suppressed_code_skeleton,
            },
        )

    def reason(self, text: str) -> str | None:
        normalized = canonical_text(text)
        if hashlib.sha256(normalized.encode()).hexdigest() in self.exact:
            return "benchmark_exact"
        if _matches_one_holdout_group(
            self.ngram_postings,
            ngram_hashes(normalized, self.ngram_size),
            self.minimum_matching_ngrams,
            self.match_fraction,
        ):
            return "benchmark_ngram"
        if looks_like_code(text):
            if _matches_one_holdout_group(
                self.code_ngram_postings,
                code_ngram_hashes(text, self.code_ngram_size),
                self.minimum_code_matching_ngrams,
                self.match_fraction,
            ):
                return "benchmark_code_ngram"
            if _matches_one_holdout_group(
                self.code_skeleton_ngram_postings,
                code_skeleton_ngram_hashes(text, self.code_skeleton_ngram_size),
                self.minimum_code_skeleton_matching_ngrams,
                self.match_fraction,
            ):
                return "benchmark_code_skeleton_ngram"
        if _matches_one_holdout_group(
            self.short_ngram_postings,
            ngram_hashes(normalized, self.short_ngram_size),
            self.minimum_short_matching_ngrams,
            self.match_fraction,
        ):
            return "benchmark_short_ngram"
        return None

    def filter(self, records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        for record in records:
            reason = self.reason(str(record["text"]))
            (removed if reason else kept).append({**record, **({"decontamination_reason": reason} if reason else {})})
        return kept, removed
