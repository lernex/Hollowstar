from __future__ import annotations

import json
import hashlib
import re
import struct
from pathlib import Path
from typing import Any

from .decontaminate import ContaminationIndex


def build_regex_word_tokenizer() -> Any:
    """Return a dependency-light tokenizer for multilingual/code MinHash shingles."""
    from datatrove.utils.word_tokenizers import WordTokenizer

    class MetisRegexWordTokenizer(WordTokenizer):
        _word = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        _sentence = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)", re.UNICODE)

        def word_tokenize(self, text: str) -> list[str]:
            return self._word.findall(text)

        def sent_tokenize(self, text: str) -> list[str]:
            return [match.group(0).strip() for match in self._sentence.finditer(text) if match.group(0).strip()]

        def span_tokenize(self, text: str) -> list[tuple[int, int]]:
            return [(match.start(), match.end()) for match in self._sentence.finditer(text)]

    return MetisRegexWordTokenizer()


def build_priority_minhash_removals(
    duplicate_folder: str | Path,
    output_folder: str | Path,
    document_folder: str | Path,
    *,
    total_tasks: int,
) -> dict[str, int]:
    """Resolve MinHash components while keeping the highest-priority document.

    DataTrove's stock clusterer keeps a union-find root, which is deterministic
    but quality-agnostic.  Metis first discovers the same connected components,
    then scans only component members and chooses by record priority and stable
    document id before writing DataTrove-compatible `.remove` files.
    """

    from datatrove.pipeline.readers import JsonlReader

    duplicates = Path(duplicate_folder)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    parent_map: dict[tuple[int, int], tuple[int, int]] = {}
    sizes: dict[tuple[int, int], int] = {}

    def parent(node: tuple[int, int]) -> tuple[int, int]:
        root = parent_map.get(node)
        if root is None:
            parent_map[node] = node
            return node
        if root != node:
            parent_map[node] = parent(root)
        return parent_map[node]

    def union(left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root = parent(left)
        right_root = parent(right)
        if left_root == right_root:
            return
        left_size = sizes.get(left_root, 1)
        right_size = sizes.get(right_root, 1)
        if left_size < right_size or (left_size == right_size and right_root < left_root):
            left_root, right_root = right_root, left_root
            left_size, right_size = right_size, left_size
        parent_map[right_root] = left_root
        sizes[left_root] = left_size + right_size
        sizes.pop(right_root, None)

    duplicate_pairs = 0
    for path in sorted(duplicates.glob("**/*.dups")):
        payload = path.read_bytes()
        if len(payload) % 16:
            raise RuntimeError(f"Corrupt MinHash duplicate file: {path}")
        for f1, d1, f2, d2 in struct.iter_unpack("<4I", payload):
            union((int(f1), int(d1)), (int(f2), int(d2)))
            duplicate_pairs += 1

    winners: dict[tuple[int, int], tuple[tuple[int, str], tuple[int, int]]] = {}
    reader = JsonlReader(str(document_folder), shuffle_files=False)
    for rank in range(total_tasks):
        for document_index, document in enumerate(reader.run(rank=rank, world_size=total_tasks)):
            node = (rank, document_index)
            if node not in parent_map:
                continue
            root = parent(node)
            stable_tie_break = hashlib.sha256(str(document.id).encode("utf-8")).hexdigest()
            candidate = (int(document.metadata.get("priority", 1)), stable_tie_break)
            previous = winners.get(root)
            if previous is None or candidate > previous[0]:
                winners[root] = (candidate, node)

    removals: dict[int, list[int]] = {}
    for node in parent_map:
        root = parent(node)
        winner = winners.get(root)
        if winner is None:
            raise RuntimeError(f"MinHash component member has no corresponding document: {node}")
        if node != winner[1]:
            removals.setdefault(node[0], []).append(node[1])
    for rank, document_ids in removals.items():
        with (output / f"{rank:06d}.remove").open("wb") as handle:
            for document_id in sorted(set(document_ids)):
                handle.write(struct.pack("<I", document_id))
    return {
        "duplicate_pairs": duplicate_pairs,
        "component_members": len(parent_map),
        "components": len(winners),
        "removed": sum(len(set(values)) for values in removals.values()),
    }


def save_contamination_index(index: ContaminationIndex, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": "metis.contamination-index/v1",
                "exact": sorted(index.exact),
                "ngrams": sorted(index.ngrams),
                "ngram_size": index.ngram_size,
                "minimum_matching_ngrams": index.minimum_matching_ngrams,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def load_contamination_index(path: str | Path) -> ContaminationIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ContaminationIndex(
        exact=frozenset(payload["exact"]),
        ngrams=frozenset(int(value) for value in payload["ngrams"]),
        ngram_size=int(payload["ngram_size"]),
        minimum_matching_ngrams=int(payload["minimum_matching_ngrams"]),
    )


def build_datatrove_decontamination_filter(index_path: str | Path) -> Any:
    from datatrove.pipeline.filters.base_filter import BaseFilter

    class MetisDecontaminationFilter(BaseFilter):
        name = "Metis benchmark decontamination"
        type = "DECONT"

        def __init__(self) -> None:
            super().__init__()
            self.index = load_contamination_index(index_path)

        def filter(self, doc: Any) -> bool | tuple[bool, str]:
            reason = self.index.reason(str(doc.text))
            if reason:
                doc.metadata["decontamination_reason"] = reason
                return False, reason
            return True

    return MetisDecontaminationFilter()
