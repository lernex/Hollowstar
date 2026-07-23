from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import zstandard as zstd

from .manifest import PHASES
from .replacement import allocate_replacements
from .state import atomic_json, utc_now


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def hamilton_apportion(total: int, weights: dict[str, int]) -> dict[str, int]:
    positive = {key: value for key, value in weights.items() if value > 0}
    denominator = sum(positive.values())
    if total == 0:
        return {key: 0 for key in weights}
    if denominator <= 0:
        raise ValueError("Cannot apportion a positive total over zero weights")
    floors: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    for key, weight in positive.items():
        numerator = total * weight
        floors[key] = numerator // denominator
        remainders.append((numerator % denominator, key))
    missing = total - sum(floors.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:missing]:
        floors[key] += 1
    return {key: floors.get(key, 0) for key in weights}


def replay_quotas(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    sources = manifest["sources"]
    phase_b_weights = {source["id"]: int(source["phase_tokens"].get("phase_b", 0)) for source in sources}
    phase_b_total = int(manifest["schedule"]["phases"]["phase_b"]["replay_tokens"])
    phase_b = hamilton_apportion(phase_b_total, phase_b_weights)
    return {
        source["id"]: {
            "phase_a": 0,
            "phase_b": phase_b[source["id"]],
            "phase_c": int(source["phase_tokens"].get("phase_c", 0)),
        }
        for source in sources
    }


def unique_quotas(manifest: dict[str, Any], replay: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        source["id"]: {
            phase: int(source["phase_tokens"].get(phase, 0)) - int(replay[source["id"]].get(phase, 0))
            for phase in PHASES
        }
        for source in manifest["sources"]
    }


def _stable_fraction(source_id: str, doc_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}\0{source_id}\0{doc_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class _AppendPool:
    def __init__(self, maximum_open: int = 32) -> None:
        self.maximum_open = maximum_open
        self.handles: OrderedDict[Path, io.TextIOWrapper] = OrderedDict()

    def write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.handles.pop(path, None)
        if handle is None:
            raw = path.open("ab")
            compressed = zstd.ZstdCompressor(level=5).stream_writer(raw, closefd=True)
            handle = io.TextIOWrapper(compressed, encoding="utf-8")
        self.handles[path] = handle
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        if len(self.handles) > self.maximum_open:
            _, oldest = self.handles.popitem(last=False)
            oldest.close()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


@dataclass
class ScheduleShard:
    phase: str
    phase_index: int
    global_index: int
    target_tokens: int
    path: Path
    written_tokens: int = 0


class ScheduleWriter:
    def __init__(self, root: Path, phase_targets: dict[str, int], shard_tokens: int) -> None:
        self.root = root
        self.pool = _AppendPool(maximum_open=48)
        self.shards: dict[str, list[ScheduleShard]] = {}
        global_index = 0
        for phase in PHASES:
            remaining = phase_targets[phase]
            phase_shards: list[ScheduleShard] = []
            phase_index = 0
            while remaining:
                target = min(shard_tokens, remaining)
                path = root / phase.replace("_", "-") / f"shard-{phase_index:05d}.jsonl.zst"
                path.unlink(missing_ok=True)
                phase_shards.append(ScheduleShard(phase, phase_index, global_index, target, path))
                remaining -= target
                phase_index += 1
                global_index += 1
            self.shards[phase] = phase_shards

    def write(
        self,
        phase: str,
        record: dict[str, Any],
        token_count: int,
        *,
        replay: bool,
        token_start: int = 0,
        exposure: int = 0,
    ) -> None:
        remaining = token_count
        offset = token_start
        seed = int.from_bytes(hashlib.sha256(f"{record['source_id']}\0{record['doc_id']}\0{int(replay)}".encode()).digest()[:8], "big")
        phase_shards = self.shards[phase]
        start = seed % len(phase_shards)
        while remaining:
            target_shard = None
            for step in range(len(phase_shards)):
                candidate = phase_shards[(start + step) % len(phase_shards)]
                if candidate.written_tokens < candidate.target_tokens:
                    target_shard = candidate
                    break
            if target_shard is None:
                raise RuntimeError(f"Phase {phase} schedule is already full")
            take = min(remaining, target_shard.target_tokens - target_shard.written_tokens)
            self.pool.write(
                target_shard.path,
                {
                    "source_id": record["source_id"],
                    "quota_source_id": record.get("quota_source_id", record["source_id"]),
                    "replacement_for_source_id": record.get("replacement_for_source_id"),
                    "replacement": bool(record.get("replacement", False)),
                    "doc_id": record["doc_id"],
                    "text": record["text"],
                    "token_start": offset,
                    "token_count": take,
                    "replay": replay,
                    "exposure": exposure,
                    "generated": bool(record.get("generated", False)),
                    "transformed": bool(record.get("transformed", False)),
                    "content_sha256": record.get("content_sha256"),
                    "text_sha256": record.get("text_sha256"),
                    "license": record.get("license"),
                    "license_status": record.get("license_status"),
                },
            )
            target_shard.written_tokens += take
            remaining -= take
            offset += take

    def close(self) -> list[dict[str, Any]]:
        self.pool.close()
        rows = []
        for phase in PHASES:
            for shard in self.shards[phase]:
                if shard.written_tokens != shard.target_tokens:
                    raise RuntimeError(
                        f"Selected shard {shard.path} has {shard.written_tokens:,} tokens, expected {shard.target_tokens:,}"
                    )
                rows.append(
                    {
                        "phase": phase,
                        "phase_index": shard.phase_index,
                        "global_index": shard.global_index,
                        "target_tokens": shard.target_tokens,
                        "path": str(shard.path),
                        "size": shard.path.stat().st_size,
                        "sha256": _sha256_file(shard.path),
                    }
                )
        return rows


def _iter_zstd(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def build_selection(
    records: Iterable[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    eligible_tokens: dict[str, int],
    output_root: Path,
    shard_tokens: int,
    token_count_contract_sha256: str | None = None,
    tokenizer_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    replay = replay_quotas(manifest)
    unique = unique_quotas(manifest, replay)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_pool_root = output_root / "replay-pool"
    fallback_root = output_root / "selection-fallback"
    if replay_pool_root.exists():
        for stale in replay_pool_root.glob("*.jsonl.zst"):
            stale.unlink()
    if fallback_root.exists():
        for stale in fallback_root.glob("*.jsonl.zst"):
            stale.unlink()
    replacement_allocation = allocate_replacements(
        manifest,
        requirements=unique,
        available_tokens=eligible_tokens,
    )
    assignments = {
        source_id: [
            {**assignment, "remaining": int(assignment["tokens"])}
            for assignment in source_assignments
        ]
        for source_id, source_assignments in replacement_allocation[
            "assignments_by_actual_source"
        ].items()
    }
    required_actual = {
        source_id: sum(int(row["tokens"]) for row in rows)
        for source_id, rows in assignments.items()
    }
    thresholds = {
        source_id: min(1.0, 1.10 * target / max(1, eligible_tokens.get(source_id, 0)))
        for source_id, target in required_actual.items()
    }
    assignment_cursor = {source_id: 0 for source_id in assignments}
    replay_pool = _AppendPool(maximum_open=24)
    fallback_pool = _AppendPool(maximum_open=24)
    phase_targets = {
        phase: int(manifest["schedule"]["phases"][phase]["target_tokens"])
        for phase in PHASES
    }
    schedule = ScheduleWriter(output_root / "schedule", phase_targets, shard_tokens)
    seed = int(manifest["selection"]["seed"])
    unique_written = {source: {phase: 0 for phase in PHASES} for source in unique}
    actual_source_unique_written = {
        source: {phase: 0 for phase in PHASES} for source in unique
    }
    replay_pool_tokens = {source: 0 for source in unique}

    def consume(record: dict[str, Any]) -> None:
        source_id = str(record["source_id"])
        source_assignments = assignments.get(source_id, [])
        cursor = assignment_cursor.get(source_id, 0)
        available = int(record["token_count"])
        consumed = 0
        while cursor < len(source_assignments) and available > 0:
            assignment = source_assignments[cursor]
            needed = int(assignment["remaining"])
            if needed <= 0:
                cursor += 1
                continue
            take = min(needed, available)
            target_source_id = str(assignment["target_source_id"])
            phase = str(assignment["phase"])
            selected_record = {
                **record,
                "quota_source_id": target_source_id,
                "replacement_for_source_id": (
                    target_source_id if target_source_id != source_id else None
                ),
                "replacement": target_source_id != source_id,
            }
            schedule.write(
                phase,
                selected_record,
                take,
                replay=False,
                token_start=consumed,
                exposure=0,
            )
            unique_written[target_source_id][phase] += take
            actual_source_unique_written[source_id][phase] += take
            assignment["remaining"] -= take
            available -= take
            desired_replay_pool = sum(replay[target_source_id].values())
            if (
                take
                and desired_replay_pool
                and replay_pool_tokens[target_source_id] < desired_replay_pool
            ):
                # Replay may draw only from token spans actually emitted as
                # unique data for this immutable quota source. The actual
                # source remains explicit when a compatible donor filled it.
                replay_record = dict(selected_record)
                replay_record["token_count"] = take
                replay_record["_selection_token_start"] = consumed
                replay_pool.write(
                    replay_pool_root / f"{target_source_id}.jsonl.zst",
                    replay_record,
                )
                replay_pool_tokens[target_source_id] += take
            consumed += take
            if int(assignment["remaining"]) == 0:
                cursor += 1
        assignment_cursor[source_id] = cursor

    for record in records:
        source_id = str(record["source_id"])
        if source_id not in required_actual or assignment_cursor[source_id] >= len(
            assignments[source_id]
        ):
            continue
        if _stable_fraction(source_id, record["doc_id"], seed) > thresholds[source_id]:
            fallback_pool.write(fallback_root / f"{source_id}.jsonl.zst", record)
            continue
        consume(record)
    fallback_pool.close()

    for source_id, source_assignments in assignments.items():
        if all(int(row["remaining"]) <= 0 for row in source_assignments):
            continue
        fallback_path = fallback_root / f"{source_id}.jsonl.zst"
        if fallback_path.exists():
            for record in _iter_zstd(fallback_path):
                consume(record)
                if all(int(row["remaining"]) <= 0 for row in source_assignments):
                    break
    replay_pool.close()
    short = {
        source_id: [
            {
                "target_source_id": row["target_source_id"],
                "phase": row["phase"],
                "tokens": int(row["remaining"]),
            }
            for row in source_assignments
            if int(row["remaining"]) > 0
        ]
        for source_id, source_assignments in assignments.items()
        if any(int(row["remaining"]) > 0 for row in source_assignments)
    }
    if short:
        raise RuntimeError(
            f"Deterministic replacement selection was short of assigned targets: {short}"
        )
    for path in fallback_root.glob("*.jsonl.zst") if fallback_root.exists() else []:
        path.unlink()
    if fallback_root.exists():
        fallback_root.rmdir()

    replay_written = {source: {phase: 0 for phase in PHASES} for source in replay}
    maximum_exposures = int(manifest["selection"]["replay"]["maximum_document_exposures"])
    for source_id, quotas in replay.items():
        if sum(quotas.values()) == 0:
            continue
        pool_path = replay_pool_root / f"{source_id}.jsonl.zst"
        if not pool_path.exists():
            raise RuntimeError(f"Replay pool is missing for {source_id}")
        remaining = dict(quotas)
        for exposure in range(1, maximum_exposures):
            for record in _iter_zstd(pool_path):
                available = int(record["token_count"])
                consumed = 0
                for phase in ("phase_b", "phase_c"):
                    need = remaining[phase]
                    if need <= 0 or available <= 0:
                        continue
                    take = min(need, available)
                    schedule.write(
                        phase,
                        record,
                        take,
                        replay=True,
                        token_start=int(record.get("_selection_token_start", 0))
                        + consumed,
                        exposure=exposure,
                    )
                    replay_written[source_id][phase] += take
                    remaining[phase] -= take
                    available -= take
                    consumed += take
                if sum(remaining.values()) == 0:
                    break
            if sum(remaining.values()) == 0:
                break
        if sum(remaining.values()):
            raise RuntimeError(f"Replay cap of {maximum_exposures} exposures cannot satisfy {source_id}: {remaining}")
    shards = schedule.close()
    schedule_contract = [
        {
            key: shard[key]
            for key in (
                "phase",
                "phase_index",
                "global_index",
                "target_tokens",
                "size",
                "sha256",
            )
        }
        for shard in shards
    ]
    payload = {
        "schema": "metis.selection-release/v2",
        "created_at": utc_now(),
        "unique_quotas": unique,
        "replay_quotas": replay,
        "unique_written": unique_written,
        "actual_source_unique_written": actual_source_unique_written,
        "replay_written": replay_written,
        "unique_tokens": sum(sum(phases.values()) for phases in unique_written.values()),
        "replay_tokens": sum(sum(phases.values()) for phases in replay_written.values()),
        "maximum_document_exposures": maximum_exposures,
        "selection_seed": seed,
        "replacement_allocation": replacement_allocation,
        "replacement_tokens": int(replacement_allocation["replacement_tokens"]),
        "token_count_contract_sha256": token_count_contract_sha256,
        "tokenizer_contract": tokenizer_contract,
        "phase_tokens": phase_targets,
        "schedule_manifest_sha256": _json_sha256(schedule_contract),
        "shards": shards,
    }
    atomic_json(output_root / "SELECTION.json", payload)
    return payload
