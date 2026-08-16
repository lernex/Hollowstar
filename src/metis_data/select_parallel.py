"""Selection that uses the node it was given.

``build_selection`` streams every eligible document through one process: it
decompresses the token-count shard, parses the row, decides where the row goes,
re-serialises it and compresses it into a schedule shard. Measured on Portage
with ``py-spy`` against the live stage, that process sat at 150% of one core on
a 192-core node -- 24% decompressing input, 19% in ``orjson.dumps``, 18% in
``orjson.loads``, 14% joining output buffers and only 3% in zstd, which already
had 32 threads. It moves roughly 3.95TB of corpus text at 29 MB/s of compressed
output, which is a little over twelve hours for a 1.378TB schedule.

None of that work is ordered. The *decisions* are: a document is selected
because of how many tokens every document before it consumed, so the routing
loop has to see the corpus in one fixed order. The bytes are not. So this
splits the two.

* ``extract`` reads the token-count shards in parallel and keeps only what a
  routing decision needs -- the source, the token count, and the three hashes
  the loop would otherwise recompute. No text crosses a process boundary.
* ``plan`` replays the *same* ``build_selection`` over those text-free rows.
  It is the identical routing code, so a plan cannot disagree with the serial
  implementation about what belongs where; it records placements instead of
  writing rows.
* ``materialise`` fans back out over the input shards and writes each
  placement's row into a per-output-shard fragment, using the same
  ``schedule_row_payload`` the serial path uses.
* ``concat`` appends each shard's fragments in emission order. zstd frames
  concatenate, so this is a byte copy, and the resulting JSONL stream is
  identical to what the serial implementation would have produced.

Measured on one idle Portage node, 96 workers ran the whole read-parse-hash-
serialise-compress path at 1.65 GB/s in and 1.64 GB/s out, which puts a full
corpus pass at about 13 minutes instead of about 13 hours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .selection import (
    _stable_fraction,
    build_selection,
    schedule_row_payload,
    seal_schedule_manifest,
    shard_seed,
    _AppendPool,
    _sha256_file,
)
from .state import atomic_json, utc_now

KEY_DTYPE = np.dtype(
    [
        ("src", "<u4"),
        ("tok", "<u4"),
        ("frac", "<f8"),
        ("su", "<u8"),
        ("sr", "<u8"),
    ]
)

# ``bkind``/``bsrc``/``bexp`` are the emission bucket: the serial loop writes
# every main-pass row, then the fallback rows source by source, then the replay
# rows source by source and exposure by exposure. Fragments are named after the
# bucket so that sorting their file names reproduces that order exactly.
PLACE_DTYPE = np.dtype(
    [
        ("row", "<u4"),
        ("out", "<u2"),
        ("tstart", "<u4"),
        ("tcount", "<u4"),
        ("replay", "u1"),
        ("exposure", "u1"),
        ("quota", "<u2"),
        ("repl", "u1"),
        ("bkind", "u1"),
        ("bsrc", "<u2"),
        ("bexp", "<u2"),
    ]
)


def _keys_dir(plan_root: Path) -> Path:
    return plan_root / "keys"


def _place_dir(plan_root: Path) -> Path:
    return plan_root / "place"


def _extra_dir(plan_root: Path) -> Path:
    return plan_root / "place-extra"


def _frag_dir(plan_root: Path) -> Path:
    return plan_root / "frag"


def _stripe(count: int, tasks: int, task: int) -> list[int]:
    """The indices this task owns, every index covered exactly once."""

    if tasks <= 0 or not 0 <= task < tasks:
        raise ValueError(f"Invalid stripe {task}/{tasks}")
    return list(range(task, count, tasks))


def _block(count: int, groups: int, group: int) -> list[int]:
    """A contiguous run of indices, every index covered exactly once.

    Materialisation has to be contiguous rather than strided: fragments are
    concatenated in group order, so a group must own an unbroken run of input
    shards for the concatenation to reproduce the order the routing loop
    emitted rows in.
    """

    if groups <= 0 or not 0 <= group < groups:
        raise ValueError(f"Invalid block {group}/{groups}")
    start = (count * group) // groups
    stop = (count * (group + 1)) // groups
    return list(range(start, stop))


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------


def _extract_shard(payload: tuple[int, str, str, dict[str, int], int]) -> int:
    from .stage_runner import _iter_rows

    index, path, out_path, source_index, seed = payload
    target = Path(out_path)
    if target.is_file():
        return index
    src: list[int] = []
    tok: list[int] = []
    frac: list[float] = []
    seed_unique: list[int] = []
    seed_replay: list[int] = []
    append_src = src.append
    append_tok = tok.append
    append_frac = frac.append
    append_su = seed_unique.append
    append_sr = seed_replay.append
    unknown = len(source_index)
    for row in _iter_rows(Path(path)):
        source_id = str(row["source_id"])
        doc_id = str(row["doc_id"])
        append_src(source_index.get(source_id, unknown))
        append_tok(int(row["token_count"]))
        append_frac(_stable_fraction(source_id, doc_id, seed))
        append_su(shard_seed(source_id, doc_id, False))
        append_sr(shard_seed(source_id, doc_id, True))
    keys = np.empty(len(src), dtype=KEY_DTYPE)
    keys["src"] = src
    keys["tok"] = tok
    keys["frac"] = frac
    keys["su"] = seed_unique
    keys["sr"] = seed_replay
    scratch = target.with_suffix(".npy.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with scratch.open("wb") as handle:
        np.save(handle, keys, allow_pickle=False)
    scratch.replace(target)
    return index


def extract(
    shard_paths: Sequence[Path],
    plan_root: Path,
    *,
    source_index: Mapping[str, int],
    seed: int,
    indices: Sequence[int],
    workers: int,
) -> None:
    keys_root = _keys_dir(plan_root)
    keys_root.mkdir(parents=True, exist_ok=True)
    payloads = [
        (
            index,
            str(shard_paths[index]),
            str(keys_root / f"{index:06d}.npy"),
            dict(source_index),
            int(seed),
        )
        for index in indices
    ]
    if not payloads:
        return
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(payloads)))) as pool:
        for _ in pool.map(_extract_shard, payloads, chunksize=1):
            pass


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


class SelectionPlanner:
    """Records where the routing loop sent each row instead of writing it."""

    def __init__(self, plan_root: Path, source_index: Mapping[str, int]) -> None:
        self.plan_root = plan_root
        self.source_index = dict(source_index)
        self.bkind = 0
        self.bsrc = 0
        self.bexp = 0
        self._shard = -1
        self._main: list[tuple] = []
        self._extra: dict[int, list[tuple]] = {}
        _place_dir(plan_root).mkdir(parents=True, exist_ok=True)
        _extra_dir(plan_root).mkdir(parents=True, exist_ok=True)

    def begin_bucket(self, kind: int, source_order: int, exposure: int) -> None:
        self.bkind = int(kind)
        self.bsrc = int(source_order)
        self.bexp = int(exposure)

    def begin_input_shard(self, index: int) -> None:
        if index == self._shard:
            return
        self._flush_main()
        self._shard = index
        self._main = []

    def _flush_main(self) -> None:
        if self._shard < 0:
            return
        _write_places(_place_dir(self.plan_root) / f"{self._shard:06d}.npy", self._main)
        self._main = []

    def emit(
        self,
        shard: Any,
        record: Mapping[str, Any],
        *,
        token_start: int,
        token_count: int,
        replay: bool,
        exposure: int,
    ) -> None:
        quota = record.get("quota_source_id") or record["source_id"]
        row = (
            record["_row"],
            shard.global_index,
            token_start,
            token_count,
            replay,
            exposure,
            self.source_index[quota],
            bool(record.get("replacement", False)),
            self.bkind,
            self.bsrc,
            self.bexp,
        )
        if self.bkind == 0:
            self._main.append(row)
            return
        self._extra.setdefault(int(record["_shard"]), []).append(row)

    def close(self) -> None:
        self._flush_main()
        self._shard = -1
        for index, rows in self._extra.items():
            _write_places(_extra_dir(self.plan_root) / f"{index:06d}.npy", rows)
        self._extra = {}


def _write_places(path: Path, rows: list[tuple]) -> None:
    if not rows:
        return
    array = np.array(rows, dtype=PLACE_DTYPE)
    scratch = path.with_suffix(".npy.tmp")
    with scratch.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    scratch.replace(path)


def _plan_records(
    plan_root: Path,
    shard_count: int,
    source_ids: Sequence[str],
    planner: SelectionPlanner,
) -> Iterator[dict[str, Any]]:
    """Feed the routing loop text-free rows in the corpus's own order.

    One dict is reused for every row. ``consume`` copies it before anything
    keeps a reference, and the only pool that stores the record itself copies
    on write, so reuse is not observable -- it just avoids allocating a billion
    short-lived dicts inside the one part of selection that cannot be spread
    across cores.
    """

    keys_root = _keys_dir(plan_root)
    record: dict[str, Any] = {
        "source_id": "",
        "doc_id": "",
        "text": "",
        "token_count": 0,
        "_frac": 0.0,
        "_shard_seed_unique": 0,
        "_shard_seed_replay": 0,
        "_shard": 0,
        "_row": 0,
    }
    for index in range(shard_count):
        keys = np.load(keys_root / f"{index:06d}.npy", allow_pickle=False)
        planner.begin_input_shard(index)
        record["_shard"] = index
        columns = zip(
            keys["src"].tolist(),
            keys["tok"].tolist(),
            keys["frac"].tolist(),
            keys["su"].tolist(),
            keys["sr"].tolist(),
        )
        for row, (src, tok, frac, seed_unique, seed_replay) in enumerate(columns):
            record["source_id"] = source_ids[src]
            record["token_count"] = tok
            record["_frac"] = frac
            record["_shard_seed_unique"] = seed_unique
            record["_shard_seed_replay"] = seed_replay
            record["_row"] = row
            yield record


def plan(
    *,
    plan_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    eligible_tokens: dict[str, int],
    shard_tokens: int,
    shard_count: int,
    source_ids: Sequence[str],
    token_count_contract_sha256: str,
    tokenizer_contract: dict[str, Any],
) -> dict[str, Any]:
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    planner = SelectionPlanner(plan_root, source_index)
    payload = build_selection(
        _plan_records(plan_root, shard_count, source_ids, planner),
        manifest=manifest,
        eligible_tokens=eligible_tokens,
        output_root=output_root,
        shard_tokens=shard_tokens,
        token_count_contract_sha256=token_count_contract_sha256,
        tokenizer_contract=tokenizer_contract,
        planner=planner,
    )
    atomic_json(plan_root / "PLAN.json", payload)
    return payload


# --------------------------------------------------------------------------
# materialise
# --------------------------------------------------------------------------


def _load_places(plan_root: Path, index: int) -> np.ndarray:
    parts = []
    for root in (_place_dir(plan_root), _extra_dir(plan_root)):
        path = root / f"{index:06d}.npy"
        if path.is_file():
            parts.append(np.load(path, allow_pickle=False))
    if not parts:
        return np.empty(0, dtype=PLACE_DTYPE)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def _materialise_group(
    payload: tuple[int, list[tuple[int, str]], str, list[str], int]
) -> int:
    # Every group already owns its whole node's worth of siblings, so a threaded
    # compressor per worker would oversubscribe the allocation many times over.
    os.environ["METIS_ZSTD_THREADS"] = "0"
    from .stage_runner import _iter_rows

    group, shards, plan_root_str, source_ids, buffer_bytes = payload
    plan_root = Path(plan_root_str)
    frag_root = _frag_dir(plan_root)
    pool = _AppendPool(
        maximum_open=48,
        flush_bytes=8 * 1024 * 1024,
        buffered_bytes=int(buffer_bytes),
    )
    suffix = f"{group:05d}.zst"
    for index, path in shards:
        places = _load_places(plan_root, index)
        if not len(places):
            continue
        order = np.argsort(places["row"], kind="stable")
        rows = places[order].tolist()
        total = len(rows)
        pointer = 0
        for row_index, record in enumerate(_iter_rows(Path(path))):
            if pointer >= total or rows[pointer][0] != row_index:
                continue
            while pointer < total and rows[pointer][0] == row_index:
                (
                    _,
                    out,
                    token_start,
                    token_count,
                    replay,
                    exposure,
                    quota,
                    replacement,
                    bkind,
                    bsrc,
                    bexp,
                ) = rows[pointer]
                quota_source_id = source_ids[quota]
                pool.write(
                    frag_root
                    / f"{out:05d}"
                    / f"{bkind}-{bsrc:05d}-{bexp:05d}-{suffix}",
                    schedule_row_payload(
                        {
                            **record,
                            "quota_source_id": quota_source_id,
                            "replacement_for_source_id": (
                                quota_source_id if replacement else None
                            ),
                            "replacement": bool(replacement),
                        },
                        token_start=int(token_start),
                        token_count=int(token_count),
                        replay=bool(replay),
                        exposure=int(exposure),
                    ),
                )
                pointer += 1
    pool.close()
    return group


def materialise(
    shard_paths: Sequence[Path],
    plan_root: Path,
    *,
    source_ids: Sequence[str],
    groups: Sequence[int],
    group_count: int,
    workers: int,
    buffer_bytes: int = 2 * 1024 * 1024 * 1024,
) -> None:
    payloads = []
    for group in groups:
        members = [
            (index, str(shard_paths[index]))
            for index in _block(len(shard_paths), group_count, group)
        ]
        if not members:
            continue
        payloads.append(
            (group, members, str(plan_root), list(source_ids), int(buffer_bytes))
        )
    if not payloads:
        return
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(payloads)))) as pool:
        for _ in pool.map(_materialise_group, payloads, chunksize=1):
            pass


# --------------------------------------------------------------------------
# concat
# --------------------------------------------------------------------------


def _concat_shard(payload: tuple[int, str, str]) -> tuple[int, int, str]:
    global_index, frag_dir, target = payload
    fragments = sorted(Path(frag_dir).glob("*.zst"))
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    scratch = path.with_suffix(".zst.tmp")
    with scratch.open("wb") as out:
        for fragment in fragments:
            with fragment.open("rb") as handle:
                while chunk := handle.read(16 * 1024 * 1024):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    scratch.replace(path)
    return global_index, size, digest.hexdigest()


def concat(
    shards: Sequence[Mapping[str, Any]],
    plan_root: Path,
    *,
    indices: Sequence[int],
    workers: int,
) -> list[dict[str, Any]]:
    frag_root = _frag_dir(plan_root)
    payloads = [
        (
            int(shards[index]["global_index"]),
            str(frag_root / f"{int(shards[index]['global_index']):05d}"),
            str(shards[index]["path"]),
        )
        for index in indices
    ]
    if not payloads:
        return []
    measured: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(payloads)))) as pool:
        for global_index, size, sha256 in pool.map(_concat_shard, payloads, chunksize=1):
            measured.append(
                {"global_index": global_index, "size": size, "sha256": sha256}
            )
    return measured


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


UNKNOWN_SOURCE = "\x00unassigned"


def source_table(manifest: Mapping[str, Any]) -> list[str]:
    """Quota sources in a fixed order, plus a slot for anything unquoted.

    ``extract`` has to encode a source as an integer without knowing whether
    the corpus still carries rows for sources the manifest dropped. Those rows
    are skipped by the routing loop anyway, so they land in a trailing slot
    that can never match a quota.
    """

    return [str(source["id"]) for source in manifest["sources"]] + [UNKNOWN_SOURCE]


def build_selection_parallel(
    shard_paths: Sequence[Path],
    *,
    manifest: dict[str, Any],
    eligible_tokens: dict[str, int],
    output_root: Path,
    shard_tokens: int,
    token_count_contract_sha256: str | None = None,
    tokenizer_contract: dict[str, Any] | None = None,
    plan_root: Path | None = None,
    workers: int = 0,
    group_count: int = 0,
    buffer_bytes: int = 2 * 1024 * 1024 * 1024,
) -> dict[str, Any]:
    """Run every pass in one process, for tests and single-node selection."""

    plan_root = plan_root or (output_root / "plan")
    workers = workers or max(1, (os.cpu_count() or 8))
    group_count = group_count or max(1, min(len(shard_paths), workers * 4))
    sources = source_table(manifest)
    source_index = {source_id: index for index, source_id in enumerate(sources)}
    seed = int(manifest["selection"]["seed"])
    extract(
        shard_paths,
        plan_root,
        source_index=source_index,
        seed=seed,
        indices=range(len(shard_paths)),
        workers=workers,
    )
    payload = plan(
        plan_root=plan_root,
        output_root=output_root,
        manifest=manifest,
        eligible_tokens=eligible_tokens,
        shard_tokens=shard_tokens,
        shard_count=len(shard_paths),
        source_ids=sources,
        token_count_contract_sha256=token_count_contract_sha256,
        tokenizer_contract=tokenizer_contract,
    )
    materialise(
        shard_paths,
        plan_root,
        source_ids=sources,
        groups=range(group_count),
        group_count=group_count,
        workers=workers,
        buffer_bytes=buffer_bytes,
    )
    measured = concat(
        payload["shards"],
        plan_root,
        indices=range(len(payload["shards"])),
        workers=workers,
    )
    return seal(payload, measured, output_root)


def seal(
    payload: dict[str, Any],
    measured: Iterable[Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    by_index = {int(row["global_index"]): row for row in measured}
    shards = []
    for shard in payload["shards"]:
        global_index = int(shard["global_index"])
        if global_index not in by_index:
            raise RuntimeError(f"Selection shard {global_index} was never materialised")
        row = by_index[global_index]
        shards.append(
            {
                **shard,
                "size": int(row["size"]),
                "sha256": str(row["sha256"]),
            }
        )
    if len(by_index) != len(shards):
        raise RuntimeError(
            f"Materialised {len(by_index)} shards for a schedule of {len(shards)}"
        )
    sealed = {
        **payload,
        "created_at": utc_now(),
        "shards": shards,
    }
    sealed["schedule_manifest_sha256"] = seal_schedule_manifest(shards)
    atomic_json(output_root / "SELECTION.json", sealed)
    return sealed


def _selection_inputs(profile: Mapping[str, Any]) -> tuple[Path, list[Path], dict[str, int], str, dict[str, Any], dict[str, Any]]:
    from .stage_runner import _manifest, sha256_file

    root = Path(profile["storage"]["lustre_root"])
    directories = profile["storage"]["directories"]
    output_root = root / directories["selected"]
    contract_path = output_root / "TOKEN_COUNT_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    token_root = root / directories["token_counts"]
    shard_paths = [
        token_root / str(task["output"]["path"])
        for task in sorted(contract["tasks"], key=lambda row: int(row["task_index"]))
    ]
    eligible: dict[str, int] = {}
    for task in contract["tasks"]:
        for source_id, tokens in task["source_tokens"].items():
            eligible[source_id] = eligible.get(source_id, 0) + int(tokens)
    return (
        output_root,
        shard_paths,
        eligible,
        sha256_file(contract_path),
        contract,
        _manifest(profile),
    )


def _task_identity(tasks: int | None, task: int | None) -> tuple[int, int]:
    if tasks is None:
        tasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if task is None:
        task = int(os.environ.get("SLURM_PROCID", "0") or 0)
    return int(tasks), int(task)


def main(argv: Sequence[str] | None = None) -> int:
    from .config import load_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("extract", "plan", "materialise", "concat", "seal"),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--buffer-gib", type=float, default=2.0)
    args = parser.parse_args(argv)

    profile = load_profile(Path(args.profile))
    (
        output_root,
        shard_paths,
        eligible,
        contract_sha256,
        contract,
        manifest,
    ) = _selection_inputs(profile)
    plan_root = output_root / "plan"
    plan_root.mkdir(parents=True, exist_ok=True)
    sources = source_table(manifest)
    workers = args.workers or max(1, (os.cpu_count() or 8))
    tasks, task = _task_identity(args.tasks, args.task)
    started = time.time()

    if args.phase == "extract":
        extract(
            shard_paths,
            plan_root,
            source_index={
                source_id: index for index, source_id in enumerate(sources)
            },
            seed=int(manifest["selection"]["seed"]),
            indices=_stripe(len(shard_paths), tasks, task),
            workers=workers,
        )
    elif args.phase == "plan":
        missing = [
            index
            for index in range(len(shard_paths))
            if not (_keys_dir(plan_root) / f"{index:06d}.npy").is_file()
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} token-count shards were never extracted, first {missing[:5]}"
            )
        plan(
            plan_root=plan_root,
            output_root=output_root,
            manifest=manifest,
            eligible_tokens=eligible,
            shard_tokens=int(profile["storage"]["final_shard_tokens"]),
            shard_count=len(shard_paths),
            source_ids=sources,
            token_count_contract_sha256=contract_sha256,
            tokenizer_contract=contract["tokenizer_contract"],
        )
    elif args.phase == "materialise":
        materialise(
            shard_paths,
            plan_root,
            source_ids=sources,
            groups=_stripe(int(args.groups), tasks, task),
            group_count=int(args.groups),
            workers=workers,
            buffer_bytes=int(args.buffer_gib * 1024 * 1024 * 1024),
        )
    elif args.phase == "concat":
        payload = json.loads((plan_root / "PLAN.json").read_text(encoding="utf-8"))
        measured = concat(
            payload["shards"],
            plan_root,
            indices=_stripe(len(payload["shards"]), tasks, task),
            workers=workers,
        )
        measure_root = plan_root / "measure"
        measure_root.mkdir(parents=True, exist_ok=True)
        atomic_json(measure_root / f"{task:05d}.json", {"shards": measured})
    else:
        payload = json.loads((plan_root / "PLAN.json").read_text(encoding="utf-8"))
        measured: list[Mapping[str, Any]] = []
        for path in sorted((plan_root / "measure").glob("*.json")):
            measured.extend(
                json.loads(path.read_text(encoding="utf-8"))["shards"]
            )
        sealed = seal(payload, measured, output_root)
        from .stage_runner import _paths

        _, state = _paths(profile)
        state.complete("select", "task-000000", sealed)
        print(
            f"sealed {len(sealed['shards'])} shards, "
            f"{sealed['unique_tokens'] + sealed['replay_tokens']:,} tokens"
        )
    print(f"phase={args.phase} task={task}/{tasks} wall={time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
