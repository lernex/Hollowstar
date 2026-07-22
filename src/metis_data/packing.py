from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tokenizers import Tokenizer

from .state import atomic_json, utc_now


@dataclass
class _Shard:
    phase: str
    index: int
    root: Path
    max_tokens: int

    def __post_init__(self) -> None:
        self.tokens: list[int] = []
        self.documents: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return self.max_tokens - len(self.tokens)

    def add(self, ids: list[int], record: dict[str, Any], *, cropped: bool) -> None:
        start = len(self.tokens)
        self.tokens.extend(ids)
        self.documents.append(
            {
                "start": start,
                "end": len(self.tokens),
                "source_id": record.get("source_id"),
                "doc_id": record.get("doc_id"),
                "replay": bool(record.get("replay", False)),
                "cropped": cropped,
            }
        )

    def write(self) -> dict[str, Any]:
        phase_root = self.root / self.phase
        phase_root.mkdir(parents=True, exist_ok=True)
        stem = f"shard-{self.index:05d}"
        binary_path = phase_root / f"{stem}.bin"
        index_path = phase_root / f"{stem}.index.jsonl"
        values = np.asarray(self.tokens, dtype=np.uint16)
        values.tofile(binary_path)
        with index_path.open("w", encoding="utf-8") as handle:
            for document in self.documents:
                handle.write(json.dumps(document, sort_keys=True) + "\n")
        binary_sha = hashlib.sha256(binary_path.read_bytes()).hexdigest()
        index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
        return {
            "phase": self.phase,
            "index": self.index,
            "tokens": len(self.tokens),
            "documents": len(self.documents),
            "binary": str(binary_path),
            "binary_sha256": binary_sha,
            "index": str(index_path),
            "index_sha256": index_sha,
        }


def pack_release(
    records: Iterable[dict[str, Any]],
    *,
    tokenizer_path: str | Path,
    output_root: str | Path,
    phase_targets: dict[str, int],
    shard_tokens: int,
    eos_token: str = "<|endoftext|>",
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() > 65_536:
        raise RuntimeError("Tokenizer vocabulary exceeds uint16 capacity")
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise RuntimeError(f"Tokenizer does not contain EOS token {eos_token!r}")
    root = Path(output_root)
    totals = {phase: 0 for phase in phase_targets}
    shards: list[dict[str, Any]] = []
    active: dict[str, _Shard] = {phase: _Shard(phase, 0, root, min(shard_tokens, target)) for phase, target in phase_targets.items()}

    for record in records:
        phase = str(record["phase"])
        if phase not in phase_targets or totals[phase] >= phase_targets[phase]:
            continue
        ids = tokenizer.encode(str(record["text"]), add_special_tokens=False).ids + [eos_id]
        while ids and totals[phase] < phase_targets[phase]:
            shard = active[phase]
            phase_remaining = phase_targets[phase] - totals[phase]
            take = min(len(ids), shard.remaining, phase_remaining)
            if take <= 0:
                shards.append(shard.write())
                active[phase] = _Shard(phase, shard.index + 1, root, min(shard_tokens, phase_remaining))
                continue
            chunk = ids[:take]
            cropped = take < len(ids)
            shard.add(chunk, record, cropped=cropped)
            totals[phase] += take
            ids = ids[take:]
            if shard.remaining == 0:
                shards.append(shard.write())
                if totals[phase] < phase_targets[phase]:
                    active[phase] = _Shard(
                        phase,
                        shard.index + 1,
                        root,
                        min(shard_tokens, phase_targets[phase] - totals[phase]),
                    )
    for phase, shard in active.items():
        if shard.tokens and (not shards or str(shard.root / phase / f"shard-{shard.index:05d}.bin") not in {item["binary"] for item in shards}):
            shards.append(shard.write())
    if totals != phase_targets:
        raise RuntimeError(f"Exact phase targets were not met: got {totals}, expected {phase_targets}")
    release = {
        "schema": "metis.data-release/v1",
        "release": root.name,
        "created_at": utc_now(),
        "token_dtype": "uint16",
        "tokenizer_sha256": hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        "phase_tokens": totals,
        "target_tokens": sum(totals.values()),
        "shards": shards,
    }
    atomic_json(root / "RELEASE.json", release)
    return release

