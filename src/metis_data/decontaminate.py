from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .dedup import canonical_text


WORD_RE = re.compile(r"\w+", re.UNICODE)


def ngram_hashes(text: str, size: int = 13) -> set[int]:
    words = WORD_RE.findall(canonical_text(text))
    if len(words) < size:
        return set()
    return {
        int.from_bytes(hashlib.blake2b(" ".join(words[index : index + size]).encode(), digest_size=8).digest(), "little")
        for index in range(len(words) - size + 1)
    }


@dataclass(frozen=True)
class ContaminationIndex:
    exact: frozenset[str]
    ngrams: frozenset[int]
    ngram_size: int = 13
    minimum_matching_ngrams: int = 2

    @classmethod
    def build(cls, holdouts: Iterable[str], *, ngram_size: int = 13, minimum_matching_ngrams: int = 2) -> "ContaminationIndex":
        exact: set[str] = set()
        ngrams: set[int] = set()
        for text in holdouts:
            normalized = canonical_text(text)
            exact.add(hashlib.sha256(normalized.encode()).hexdigest())
            ngrams.update(ngram_hashes(normalized, ngram_size))
        return cls(frozenset(exact), frozenset(ngrams), ngram_size, minimum_matching_ngrams)

    def reason(self, text: str) -> str | None:
        normalized = canonical_text(text)
        if hashlib.sha256(normalized.encode()).hexdigest() in self.exact:
            return "benchmark_exact"
        matches = len(ngram_hashes(normalized, self.ngram_size) & self.ngrams)
        if matches >= self.minimum_matching_ngrams:
            return "benchmark_ngram"
        return None

    def filter(self, records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        for record in records:
            reason = self.reason(str(record["text"]))
            (removed if reason else kept).append({**record, **({"decontamination_reason": reason} if reason else {})})
        return kept, removed

