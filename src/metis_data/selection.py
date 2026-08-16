from __future__ import annotations

import hashlib
import io
import json
from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import zstandard as zstd

from .manifest import PHASES
from .replacement import allocate_replacements
from .state import atomic_json, utc_now, zstd_bulk_compressor


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


try:  # orjson serialises these rows several times faster than the stdlib
    import orjson as _orjson

    def _dumps_line(payload: dict[str, Any]) -> bytes:
        return _orjson.dumps(payload, option=_orjson.OPT_SORT_KEYS) + b"\n"

except ImportError:  # pragma: no cover - exercised only where orjson is absent
    def _dumps_line(payload: dict[str, Any]) -> bytes:
        return (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")


class _AppendPool:
    """Batch rows per shard, then append one large frame per flush.

    Selection routes each record to a shard by hash, so consecutive records
    land on essentially unrelated shards out of several hundred. Holding a
    compressor open per shard and evicting the oldest -- 48 handles against
    555 live shards in phase_a alone -- meant almost every write opened a
    file, built a compressor, wrote one line, and later closed it. That is the
    file-pool thrash already recorded for span dedup in the 1.7 lessons, and
    it has a second cost here: a frame holding a few rows never reaches the
    job size zstd needs to use more than one core, so the threaded compressor
    sat idle at 0.97 cores while 191 were free.

    Buffering instead means a flush is one compress() call over tens of
    megabytes, which is large enough both to amortise the open and to give the
    compressor real work to divide. Frames stay independently appended, which
    the reader already handles.

    Rows are buffered already encoded. Profiling the live stage put 30% of it
    in stdlib iterencode and another 25% in joining a 64MB string and encoding
    it, so keeping bytes end to end removes both.
    """

    def __init__(
        self,
        maximum_open: int = 32,
        *,
        flush_bytes: int = 64 * 1024 * 1024,
        buffered_bytes: int = 6 * 1024 * 1024 * 1024,
    ) -> None:
        self.maximum_open = maximum_open
        self.flush_bytes = int(flush_bytes)
        self.buffered_bytes = int(buffered_bytes)
        self.buffers: dict[Path, list[bytes]] = {}
        self.sizes: dict[Path, int] = {}
        self.buffered = 0

    def write(self, path: Path, payload: dict[str, Any]) -> None:
        line = _dumps_line(payload)
        self.buffers.setdefault(path, []).append(line)
        size = len(line)
        self.sizes[path] = self.sizes.get(path, 0) + size
        self.buffered += size
        if self.sizes[path] >= self.flush_bytes:
            self._flush(path)
        elif self.buffered >= self.buffered_bytes:
            self._flush(max(self.sizes, key=lambda key: self.sizes[key]))

    def _flush(self, path: Path) -> None:
        lines = self.buffers.pop(path, None)
        if not lines:
            self.sizes.pop(path, None)
            return
        self.buffered -= self.sizes.pop(path, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = zstd_bulk_compressor(5).compress(b"".join(lines))
        with path.open("ab") as raw:
            raw.write(frame)

    def close(self) -> None:
        for path in sorted(self.buffers, key=lambda value: str(value)):
            self._flush(path)
        self.buffers.clear()
        self.sizes.clear()
        self.buffered = 0


def schedule_row_payload(
    record: Mapping[str, Any],
    *,
    token_start: int,
    token_count: int,
    replay: bool,
    exposure: int,
) -> dict[str, Any]:
    """The one definition of a schedule row.

    Selection can either materialise rows as it routes them or route first and
    materialise later across many cores. Both paths call this, so a shard built
    either way carries identical fields; a second hand-written copy of this
    dict is exactly the drifting twin the 1.7 lessons warn about.
    """

    return {
        "source_id": record["source_id"],
        "quota_source_id": record.get("quota_source_id", record["source_id"]),
        "replacement_for_source_id": record.get("replacement_for_source_id"),
        "replacement": bool(record.get("replacement", False)),
        "doc_id": record["doc_id"],
        "text": record["text"],
        "token_start": token_start,
        "token_count": token_count,
        "replay": replay,
        "exposure": exposure,
        "generated": bool(record.get("generated", False)),
        "transformed": bool(record.get("transformed", False)),
        "content_sha256": record.get("content_sha256"),
        "text_sha256": record.get("text_sha256"),
        "license": record.get("license"),
        "license_status": record.get("license_status"),
    }


def shard_seed(source_id: str, doc_id: str, replay: bool) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{source_id}\0{doc_id}\0{int(replay)}".encode()).digest()[:8],
        "big",
    )


class _RowSink:
    """Compress selected rows into their shard as routing decides them."""

    def __init__(self) -> None:
        self.pool = _AppendPool(maximum_open=48)

    def emit(
        self,
        shard: "ScheduleShard",
        record: Mapping[str, Any],
        *,
        token_start: int,
        token_count: int,
        replay: bool,
        exposure: int,
    ) -> None:
        self.pool.write(
            shard.path,
            schedule_row_payload(
                record,
                token_start=token_start,
                token_count=token_count,
                replay=replay,
                exposure=exposure,
            ),
        )

    def close(self) -> None:
        self.pool.close()


@dataclass
class ScheduleShard:
    phase: str
    phase_index: int
    global_index: int
    target_tokens: int
    path: Path
    written_tokens: int = 0


class ScheduleWriter:
    def __init__(
        self,
        root: Path,
        phase_targets: dict[str, int],
        shard_tokens: int,
        *,
        sink: Any | None = None,
        reset: bool = True,
    ) -> None:
        self.root = root
        self.sink = sink if sink is not None else _RowSink()
        self.shards: dict[str, list[ScheduleShard]] = {}
        # Indices of the shards that still have room, per phase, kept sorted so
        # the forward scan for the first non-full shard stays logarithmic. The
        # readable form rescanned up to 555 shards per write and did it again
        # for every fragment of a straddling document.
        self._open: dict[str, list[int]] = {}
        global_index = 0
        for phase in PHASES:
            remaining = phase_targets[phase]
            phase_shards: list[ScheduleShard] = []
            phase_index = 0
            while remaining:
                target = min(shard_tokens, remaining)
                path = root / phase.replace("_", "-") / f"shard-{phase_index:05d}.jsonl.zst"
                if reset:
                    path.unlink(missing_ok=True)
                phase_shards.append(ScheduleShard(phase, phase_index, global_index, target, path))
                remaining -= target
                phase_index += 1
                global_index += 1
            self.shards[phase] = phase_shards
            self._open[phase] = list(range(len(phase_shards)))

    def _first_open(self, phase: str, start: int) -> ScheduleShard:
        """The first shard with room at or after ``start``, wrapping once."""

        open_indices = self._open[phase]
        if not open_indices:
            raise RuntimeError(f"Phase {phase} schedule is already full")
        position = bisect_left(open_indices, start)
        if position == len(open_indices):
            position = 0
        return self.shards[phase][open_indices[position]]

    def _retire(self, shard: ScheduleShard) -> None:
        open_indices = self._open[shard.phase]
        position = bisect_left(open_indices, shard.phase_index)
        if position < len(open_indices) and open_indices[position] == shard.phase_index:
            open_indices.pop(position)

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
        seed = record.get("_shard_seed_replay" if replay else "_shard_seed_unique")
        if seed is None:
            seed = shard_seed(str(record["source_id"]), str(record["doc_id"]), replay)
        start = seed % len(self.shards[phase])
        while remaining:
            target_shard = self._first_open(phase, start)
            take = min(remaining, target_shard.target_tokens - target_shard.written_tokens)
            self.sink.emit(
                target_shard,
                record,
                token_start=offset,
                token_count=take,
                replay=replay,
                exposure=exposure,
            )
            target_shard.written_tokens += take
            if target_shard.written_tokens == target_shard.target_tokens:
                self._retire(target_shard)
            remaining -= take
            offset += take

    def close(self) -> list[dict[str, Any]]:
        self.sink.close()
        return self.verify(measure=True)

    def verify(self, *, measure: bool) -> list[dict[str, Any]]:
        rows = []
        for phase in PHASES:
            for shard in self.shards[phase]:
                if shard.written_tokens != shard.target_tokens:
                    raise RuntimeError(
                        f"Selected shard {shard.path} has {shard.written_tokens:,} tokens, expected {shard.target_tokens:,}"
                    )
                row = {
                    "phase": phase,
                    "phase_index": shard.phase_index,
                    "global_index": shard.global_index,
                    "target_tokens": shard.target_tokens,
                    "path": str(shard.path),
                }
                if measure:
                    row["size"] = shard.path.stat().st_size
                    row["sha256"] = _sha256_file(shard.path)
                rows.append(row)
        return rows


def _iter_zstd(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


class _MemoryPool:
    """A pool that keeps records in memory under the same interface.

    Planning reruns the identical routing loop over records that carry no text,
    so the fallback and replay pools never need to reach Lustre. Matching
    ``_AppendPool``'s surface keeps one routing implementation rather than two.
    """

    def __init__(self) -> None:
        self.rows: dict[Path, list[dict[str, Any]]] = {}

    def write(self, path: Path, payload: dict[str, Any]) -> None:
        # Planning reuses one record dict per row, so the pool has to take its
        # own copy the way the serialising pool implicitly does.
        self.rows.setdefault(path, []).append(dict(payload))

    def exists(self, path: Path) -> bool:
        return path in self.rows

    def read(self, path: Path) -> Iterator[dict[str, Any]]:
        return iter(self.rows.get(path, ()))

    def close(self) -> None:
        return None


def build_selection(
    records: Iterable[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    eligible_tokens: dict[str, int],
    output_root: Path,
    shard_tokens: int,
    token_count_contract_sha256: str | None = None,
    tokenizer_contract: dict[str, Any] | None = None,
    planner: Any | None = None,
) -> dict[str, Any]:
    planning = planner is not None
    replay = replay_quotas(manifest)
    unique = unique_quotas(manifest, replay)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_pool_root = output_root / "replay-pool"
    fallback_root = output_root / "selection-fallback"
    if not planning:
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
    replay_pool = _MemoryPool() if planning else _AppendPool(maximum_open=24)
    fallback_pool = _MemoryPool() if planning else _AppendPool(maximum_open=24)
    phase_targets = {
        phase: int(manifest["schedule"]["phases"][phase]["target_tokens"])
        for phase in PHASES
    }
    schedule = ScheduleWriter(
        output_root / "schedule",
        phase_targets,
        shard_tokens,
        sink=planner,
        reset=not planning,
    )
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
        fraction = record.get("_frac")
        if fraction is None:
            fraction = _stable_fraction(source_id, record["doc_id"], seed)
        if fraction > thresholds[source_id]:
            fallback_pool.write(fallback_root / f"{source_id}.jsonl.zst", record)
            continue
        consume(record)
    fallback_pool.close()

    for order, (source_id, source_assignments) in enumerate(assignments.items()):
        if all(int(row["remaining"]) <= 0 for row in source_assignments):
            continue
        fallback_path = fallback_root / f"{source_id}.jsonl.zst"
        if planning:
            planner.begin_bucket(1, order, 0)
            available = fallback_pool.exists(fallback_path)
            stream = fallback_pool.read(fallback_path)
        else:
            available = fallback_path.exists()
            stream = _iter_zstd(fallback_path) if available else iter(())
        if available:
            for record in stream:
                consume(record)
                if all(int(row["remaining"]) <= 0 for row in source_assignments):
                    break
    if planning:
        planner.begin_bucket(0, 0, 0)
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
    if not planning:
        for path in fallback_root.glob("*.jsonl.zst") if fallback_root.exists() else []:
            path.unlink()
        if fallback_root.exists():
            fallback_root.rmdir()

    replay_written = {source: {phase: 0 for phase in PHASES} for source in replay}
    maximum_exposures = int(manifest["selection"]["replay"]["maximum_document_exposures"])
    for order, (source_id, quotas) in enumerate(replay.items()):
        if sum(quotas.values()) == 0:
            continue
        pool_path = replay_pool_root / f"{source_id}.jsonl.zst"
        if not (replay_pool.exists(pool_path) if planning else pool_path.exists()):
            raise RuntimeError(f"Replay pool is missing for {source_id}")
        remaining = dict(quotas)
        for exposure in range(1, maximum_exposures):
            if planning:
                planner.begin_bucket(2, order, exposure)
            stream = (
                replay_pool.read(pool_path) if planning else _iter_zstd(pool_path)
            )
            for record in stream:
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
    if planning:
        planner.close()
        shards = schedule.verify(measure=False)
    else:
        shards = schedule.close()
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
        "shards": shards,
    }
    if planning:
        # The shard bytes do not exist yet; the parallel builder seals the
        # manifest once its workers have materialised and hashed them.
        return payload
    payload["schedule_manifest_sha256"] = seal_schedule_manifest(shards)
    atomic_json(output_root / "SELECTION.json", payload)
    return payload


def seal_schedule_manifest(shards: Iterable[Mapping[str, Any]]) -> str:
    return _json_sha256(
        [
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
    )
