from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def canonical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return SPACE_RE.sub(" ", normalized).strip()


def exact_digest(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")


def simhash64(text: str, ngram_size: int = 5) -> int:
    words = WORD_RE.findall(canonical_text(text))
    if len(words) < ngram_size:
        words = words or [""]
        shingles = [" ".join(words)]
    else:
        shingles = [" ".join(words[index : index + ngram_size]) for index in range(len(words) - ngram_size + 1)]
    weights = [0] * 64
    for shingle in shingles:
        value = _hash64(shingle)
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    output = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            output |= 1 << bit
    return output


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(frozen=True)
class DedupResult:
    kept: tuple[dict[str, Any], ...]
    removed: tuple[dict[str, Any], ...]


def deduplicate_records(records: Iterable[dict[str, Any]], *, maximum_simhash_distance: int = 3) -> DedupResult:
    exact_winners: dict[str, dict[str, Any]] = {}
    removed: list[dict[str, Any]] = []
    for record in records:
        digest = exact_digest(str(record["text"]))
        record = {**record, "content_sha256": digest}
        incumbent = exact_winners.get(digest)
        if incumbent is None or int(record.get("priority", 1)) > int(incumbent.get("priority", 1)):
            if incumbent is not None:
                removed.append({**incumbent, "dedup_reason": "exact_lower_priority"})
            exact_winners[digest] = record
        else:
            removed.append({**record, "dedup_reason": "exact_lower_priority"})

    bands: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    kept: list[dict[str, Any]] = []
    for record in sorted(exact_winners.values(), key=lambda item: (-int(item.get("priority", 1)), item["content_sha256"])):
        signature = simhash64(str(record["text"]))
        candidates: list[tuple[int, dict[str, Any]]] = []
        for band in range(4):
            key = (band, (signature >> (band * 16)) & 0xFFFF)
            candidates.extend(bands[key])
        if any(hamming_distance(signature, candidate_sig) <= maximum_simhash_distance for candidate_sig, _ in candidates):
            removed.append({**record, "dedup_reason": "near_duplicate"})
            continue
        accepted = {**record, "simhash64": f"{signature:016x}"}
        kept.append(accepted)
        for band in range(4):
            key = (band, (signature >> (band * 16)) & 0xFFFF)
            bands[key].append((signature, accepted))
    return DedupResult(tuple(kept), tuple(removed))

