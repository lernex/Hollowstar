"""Bounded, restartable CPU work over sealed objects arriving from either uplink."""

from __future__ import annotations

import heapq
import json
import multiprocessing
import os
import signal
import socket
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from . import prep
from .acquisition import CapacityPending, receipt_path
from .admission import admit_source, claim
from .cli import _stop_event, append_event, code_commit, code_root, load_run, safe_error
from .common import (
    ObjectSpec, RawReceipt, atomic_json, digest_json, read_receipt,
    sha256_file, under_root, utc_now, write_receipt,
)
from .policy import policy_config


_IMPLEMENTATION_SHA256 = sha256_file(Path(__file__))


class EventTail:
    """Only complete, sealed journal lines advance the in-memory cursor."""

    def __init__(self) -> None:
        self.positions: dict[Path, tuple[int, int]] = {}

    def read(self, path: Path) -> list[dict[str, Any]]:
        stat = path.stat()
        inode, offset = self.positions.get(path, (stat.st_ino, 0))
        if inode != stat.st_ino or stat.st_size < offset:
            raise RuntimeError(f"Committed event journal was replaced or truncated: {path}")
        result = []
        with path.open("rb") as stream:
            stream.seek(offset)
            while raw := stream.readline():
                if not raw.endswith(b"\n"):
                    break
                value = json.loads(raw)
                if value.pop("event_sha256", None) != digest_json(value):
                    raise RuntimeError(f"Corrupt committed event: {path}")
                result.append(value)
                offset = stream.tell()
        self.positions[path] = (inode, offset)
        return result


def worker_configuration(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = load_run(root)
    settings = dict(run["config"]["prep"])
    profiles = (code_root() / settings["quality_profiles_path"]).resolve()
    policy = policy_config(root)
    if not policy["policy_ready"]:
        raise RuntimeError("Verified eligibility policies must be available before CPU workers start")
    config = {
        **settings, **policy, "root": str(root), "quality_profiles_path": str(profiles),
        "enforce_storage_budget": True,
    }
    extraction = digest_json({
        "code": prep._stage_code("normalization"),
        "layout": {key: config.get(key) for key in
                   ("output_chunk_bytes", "batch_size", "maximum_record_bytes")},
    })
    identity = {
        "schema": "metis17.preparation-generation/v1",
        "extraction_generation": extraction,
        "eligibility_code": prep._stage_code("chunk_eligibility"),
        "quality_profiles_sha256": sha256_file(profiles),
        "policy_inputs": {
            key: sha256_file(Path(config[key]))
            for key in ("benchmark_registry", "decontamination_index", "opt_out_snapshot")
        },
        "source_contract": run["config_sha256"],
        "dedup": run["config"]["dedup"],
    }
    generation = digest_json(identity)
    index_identity = {
        "eligibility_generation": generation,
        "code": {
            path.name: sha256_file(path)
            for path in sorted(Path(__file__).parent.glob("dedup*.py"))
        },
    }
    index_generation = digest_json(index_identity)
    config.update({
        "generation": generation, "extraction_generation": extraction,
        "index_generation": index_generation,
        "dedup": run["config"]["dedup"],
    })
    path = root / "preparation" / "generations" / f"{generation}.json"
    if path.exists() and read_receipt(path) != identity:
        raise RuntimeError("Preparation generation identity changed")
    write_receipt(path, identity)
    write_receipt(root / "preparation" / "indexes" / f"{index_generation}.json", index_identity)
    return config, run


def raw_event(root: Path, event: Mapping[str, Any]) -> tuple[ObjectSpec, RawReceipt]:
    spec = ObjectSpec.from_dict(event["spec"])
    raw = RawReceipt.from_dict(event)
    saved = read_receipt(receipt_path(root, spec.object_id))
    if (
        RawReceipt.from_dict(saved) != raw
        or saved["spec"] != spec.to_dict()
        or raw.object_id != spec.object_id
        or raw.source_id != spec.source_id
    ):
        raise RuntimeError("RAW_READY journal disagrees with its immutable object receipt")
    return spec, raw


def _normalized_directory(root: Path, config: Mapping[str, Any], spec: ObjectSpec) -> Path:
    return root / "reblock" / config["extraction_generation"] / spec.object_id[:2] / spec.object_id


def _prepared_directory(root: Path, config: Mapping[str, Any], spec: ObjectSpec) -> Path:
    return root / "prepared" / config["generation"] / spec.object_id[:2] / spec.object_id


def _object_marker(root: Path, generation: str, object_id: str) -> Path:
    return root / "state" / "prepared-objects" / generation / object_id[:2] / f"{object_id}.json"


def _failure_marker(root: Path, generation: str, object_id: str) -> Path:
    return root / "state" / "prep-failures" / generation / object_id[:2] / f"{object_id}.json"


def failure_blocks(
    value: Mapping[str, Any], limits_sha256: str, *,
    implementation_sha256: str = _IMPLEMENTATION_SHA256,
) -> bool:
    if value.get("worker_sha256") not in (None, implementation_sha256):
        return False
    return value["status"] != "capacity_pending" or value.get("limits_sha256") == limits_sha256


def observe_failure(failures: dict[str, dict[str, Any]], event: dict[str, Any]) -> None:
    previous = failures.get(event["object_id"])
    if previous is None or event["created_at"] > previous["created_at"]:
        failures[event["object_id"]] = event


def _reblock_job(spec: ObjectSpec, raw: RawReceipt, config: dict[str, Any]) -> str:
    root = Path(config["root"])
    value = prep.reblock_object(spec, raw, _normalized_directory(root, config, spec), config)
    return value["receipt_path"]


def screen_chunk(path: Path, spec: ObjectSpec, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["root"])
    value = prep.prepare_chunk(path, _prepared_directory(root, config, spec), config)
    return {
        key: value[key] for key in (
            "status", "receipt_path", "chunk_id", "input_documents",
            "accepted_documents", "eligible_documents", "rejected", "quarantined",
        )
    }


def _execute(function: Any, *args: Any) -> dict[str, Any]:
    try:
        return {"ok": True, "result": function(*args)}
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        # Source-specific exceptions have constructors that pickle cannot
        # replay. Preserve their traceback rather than losing the pool's
        # result thread while reconstructing the exception on the parent.
        config = args[-1] if args and isinstance(args[-1], Mapping) else {}
        return {
            "ok": False, "capacity_pending": isinstance(exc, CapacityPending),
            "limits_sha256": config.get("limits_sha256"), **safe_error(exc),
        }


class WorkerFailure(RuntimeError):
    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = dict(result)
        super().__init__(result["traceback"])


def _job_result(job: Any) -> Any:
    value = job.result()
    if value["ok"] is not True:
        raise WorkerFailure(value)
    return value["result"]


def _child_initialize() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_ready() -> bool:
    return True


def index_chunk(stage_path: Path, spec: ObjectSpec, config: dict[str, Any]) -> dict[str, Any]:
    from .dedup import generate_signatures, ingest_eligible

    root = Path(config["root"])
    stage_path = under_root(root, str(stage_path))
    stage = read_receipt(stage_path)
    if (
        stage["status"] != "ELIGIBLE" or stage["eligible"] is not True
        or stage["training_ready"] is not True or stage.get("object_complete") is not True
    ):
        raise RuntimeError("Only EOF-covered eligible chunks can enter deduplication")
    seal = digest_json(stage)
    marker = root / "state" / "indexed-chunks" / config["index_generation"] / seal[:2] / f"{seal}.json"
    if marker.exists():
        result = read_receipt(marker)
        if result["stage_receipt_sha256"] != seal:
            raise RuntimeError("Indexed-chunk receipt changed")
        return result
    paths = [under_root(root, chunk["path"]) for chunk in stage["chunks"]]
    exact = signatures = None
    if paths:
        kwargs: dict[str, Any] = {"stage_receipt_path": stage_path, "stage_receipt_sha256": seal}
        if config.get("enforce_storage_budget") is True:
            from .storage import WorkingBudget

            kwargs["working_budget"] = WorkingBudget(root)
        exact = ingest_eligible(
            paths, root / "dedup" / "exact" / config["index_generation"],
            batch_id=seal, bucket_count=config["dedup"]["bucket_count"], **kwargs,
        )
        metadata = spec.policy.get("metadata", {})
        crawl = metadata.get("crawl")
        snapshot = str(crawl or f"unresolved-crawl:{spec.source_id}:{spec.revision}")
        namespace = f"{spec.policy['category']}:{'generated' if spec.policy['generated'] else 'organic'}"
        signatures = generate_signatures(
            paths, root / "dedup" / "signatures" / config["index_generation"],
            batch_id=seal, snapshot=snapshot, semantic_namespace=namespace, **kwargs,
        )
    result = {
        "schema": "metis17.indexed-chunk/v1",
        "generation": config["generation"],
        "index_generation": config["index_generation"],
        "object_id": spec.object_id, "source_id": spec.source_id,
        "chunk_id": stage["chunk_id"],
        "receipt_path": str(stage_path.relative_to(root)),
        "stage_receipt_sha256": seal,
        "eligible_documents": stage["eligible_documents"],
        "exact_result_sha256": digest_json(exact) if exact is not None else None,
        "signature_result_sha256": digest_json(signatures) if signatures is not None else None,
        "near_deletion_complete": False,
        "created_at": utc_now(),
    }
    write_receipt(marker, result)
    return result


@dataclass
class ObjectWork:
    spec: ObjectSpec
    raw: RawReceipt
    lease: BinaryIO
    output: Path
    reader: Any = None
    normalized: dict[str, Any] | None = None
    ready: dict[str, Path] = field(default_factory=dict)
    ready_files: set[Path] = field(default_factory=set)
    screened: dict[str, dict[str, Any]] = field(default_factory=dict)
    indexed: set[str] = field(default_factory=set)
    pending: set[str] = field(default_factory=set)
    failed: bool = False
    admitted: bool = False


def prep_service(
    root: Path, *, workers: int = 32, raw_readers: int | None = None,
    idle_seconds: float = 600, maximum_seconds: float = 42_000,
) -> None:
    if workers < 1 or idle_seconds <= 0 or maximum_seconds <= 0:
        raise ValueError("Worker counts and service time bounds must be positive")
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError("Preparation requires a Linux/POSIX fork runtime with shared verified policy mappings")
    config, run = worker_configuration(root)
    readers = raw_readers or int(config["raw_readers_per_node"])
    if readers < 1:
        raise ValueError("A positive raw-reader count is required")
    host = socket.gethostname().split(".", 1)[0]
    owner = f"{host}-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    status_path = root / "status" / f"prep-{owner}.json"
    generation = config["generation"]
    index_generation = config["index_generation"]
    stop = _stop_event()
    tail = EventTail()
    waiting: dict[str, tuple[ObjectSpec, RawReceipt]] = {}
    observed: set[str] = set()
    closed: set[str] = set()
    completed_objects: set[str] = set()
    failures: dict[str, dict[str, Any]] = {}
    limits_sha256 = ""
    active: dict[str, ObjectWork] = {}
    jobs: dict[tuple[str, str], tuple[str, Any]] = {}
    totals: Counter[str] = Counter()
    started = last_work = time.monotonic()
    commit = code_commit()
    atomic_json(status_path, {
        "status": "loading_policies", "pid": os.getpid(), "host": host,
        "generation": generation, "code_commit": commit, "updated_at": utc_now(),
    })
    prep.prepare_runtime(config, require_ready=True)
    context = multiprocessing.get_context("fork")
    # A fork executor starts its workers before the first submission returns.
    # Warm it before claiming objects, and reserve separate reader/chunk slots
    # in one pool. Abrupt worker death then fails futures instead of hanging.
    with ProcessPoolExecutor(
        max_workers=readers + workers, mp_context=context, initializer=_child_initialize,
    ) as pool:
        pool.submit(_worker_ready).result()
        try:
            while True:
                now = time.monotonic()
                if (root / "STOP").exists() or now - started >= maximum_seconds:
                    stop.set()
                current_capacity = (
                    read_receipt(root / "limits.json")
                    if config.get("enforce_storage_budget") is True else run["config"]["limits"]
                )
                current_limits = digest_json(current_capacity)
                config["limits_sha256"] = current_limits
                for stream in ("prepared", "prep-errors"):
                    for path in sorted((root / "events" / stream).glob("*.jsonl")):
                        for event in tail.read(path):
                            if event.get("index_generation") == index_generation:
                                if stream == "prepared":
                                    completed_objects.add(event["object_id"])
                                else:
                                    observe_failure(failures, event)
                newly_closed = completed_objects | {
                    object_id for object_id, value in failures.items()
                    if failure_blocks(value, current_limits)
                }
                for object_id in newly_closed:
                    waiting.pop(object_id, None)
                if current_limits != limits_sha256:
                    for object_id in closed - newly_closed:
                        if object_id in observed and object_id not in active:
                            value = read_receipt(receipt_path(root, object_id))
                            waiting[object_id] = (ObjectSpec.from_dict(value["spec"]), RawReceipt.from_dict(value))
                    limits_sha256 = current_limits
                closed = newly_closed
                for path in sorted((root / "events" / "raw").glob("*.jsonl")):
                    for event in tail.read(path):
                        spec, raw = raw_event(root, event)
                        if spec.object_id not in observed:
                            observed.add(spec.object_id)
                            if spec.object_id not in closed:
                                waiting[spec.object_id] = (spec, raw)
                for object_id, work in list(active.items()):
                    if work.reader is not None and work.reader.done():
                        try:
                            path = _job_result(work.reader)
                            work.normalized = read_receipt(under_root(root, path))
                            totals["reblocked_objects"] += 1
                            totals["reblocked_payload_bytes"] += work.raw.byte_count
                            totals["normalized_documents"] += work.normalized["normalized_documents"]
                        except BrokenProcessPool:
                            raise
                        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                            _record_failure(root, config, work, "reblock", exc, owner)
                            totals["failed_tasks"] += 1
                        work.reader = None
                        last_work = now
                    for path in work.output.glob("normalized/*/part-*.READY.json"):
                        if path in work.ready_files:
                            continue
                        value = read_receipt(path)
                        chunk_id = value["chunk_id"]
                        work.ready_files.add(path)
                        if chunk_id not in work.ready:
                            if value["object_id"] != object_id:
                                raise RuntimeError("Chunk discovered under a different object's namespace")
                            work.ready[chunk_id] = path
                            totals["normalized_chunks"] += 1
                    if work.normalized is not None:
                        expected = {value["chunk_id"] for value in work.normalized["chunks"]}
                        if set(work.ready) != expected:
                            raise RuntimeError("Discovered chunks do not exactly cover the sealed object")
                for (object_id, chunk_id), (kind, job) in list(jobs.items()):
                    if not job.done():
                        continue
                    work = active[object_id]
                    work.pending.remove(chunk_id)
                    del jobs[(object_id, chunk_id)]
                    try:
                        value = _job_result(job)
                        if kind == "screen":
                            previous = work.screened.get(chunk_id)
                            work.screened[chunk_id] = value
                            if previous is None:
                                totals["screened_chunks"] += 1
                                totals["screened_documents"] += value["input_documents"]
                                totals["rejected_documents"] += sum(value["rejected"].values())
                                totals["quarantined_documents"] += sum(value["quarantined"].values())
                            if value["status"] == "ELIGIBLE":
                                stage = read_receipt(under_root(root, value["receipt_path"]))
                                append_event(root, f"eligible/{owner}", {
                                    "generation": generation, "object_id": object_id,
                                    "index_generation": index_generation,
                                    "source_id": work.spec.source_id, "chunk_id": chunk_id,
                                    "receipt_path": value["receipt_path"],
                                    "stage_receipt_sha256": digest_json(stage),
                                    "input_documents": value["input_documents"],
                                    "eligible_documents": value["eligible_documents"],
                                })
                                totals["eligible_documents"] += value["eligible_documents"]
                        else:
                            work.indexed.add(chunk_id)
                            totals["indexed_chunks"] += 1
                            totals["indexed_documents"] += value["eligible_documents"]
                            append_event(root, f"indexed/{owner}", value)
                    except BrokenProcessPool:
                        raise
                    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                        _record_failure(root, config, work, kind, exc, owner)
                        totals["failed_tasks"] += 1
                    last_work = now
                for object_id, work in list(active.items()):
                    if work.normalized is not None and not work.failed and not work.admitted:
                        if (
                            set(work.screened) == set(work.ready)
                            and all(row["status"] == "ELIGIBLE" for row in work.screened.values())
                        ):
                            admission = admit_source(
                                root, work.spec, work.normalized, work.screened,
                                generation=generation,
                                minimum_acceptance=float(config["source_minimum_acceptance"]),
                            )
                            append_event(root, f"admissions/{owner}", {
                                key: value for key, value in admission.items() if key != "eligible_receipts"
                            })
                            work.admitted = True
                    finished = (
                        work.normalized is not None and work.indexed == set(work.ready)
                        and not work.pending and work.reader is None and not work.failed
                    )
                    if finished:
                        result = {
                            "schema": "metis17.prepared-object-indexed/v1",
                            "generation": generation, "object_id": object_id,
                            "index_generation": index_generation,
                            "source_id": work.spec.source_id,
                            "raw_byte_count": work.raw.byte_count,
                            "input_documents": work.normalized["input_documents"],
                            "normalized_documents": work.normalized["normalized_documents"],
                            "eligible_documents": sum(row["eligible_documents"] for row in work.screened.values()),
                            "chunk_ids": sorted(work.indexed), "created_at": utc_now(),
                            "near_deletion_complete": False,
                        }
                        write_receipt(_object_marker(root, index_generation, object_id), result)
                        append_event(root, f"prepared/{owner}", {
                            **{key: value for key, value in result.items() if key != "chunk_ids"},
                            "chunk_count": len(work.indexed),
                        })
                        totals["completed_objects"] += 1
                        totals["prepared_payload_bytes"] += work.raw.byte_count
                    if finished or (work.failed and not work.pending and work.reader is None):
                        work.lease.close()
                        del active[object_id]
                        if work.failed and not stop.is_set():
                            failure = read_receipt(_failure_marker(root, index_generation, object_id))
                            if not failure_blocks(failure, current_limits):
                                waiting[object_id] = (work.spec, work.raw)
                if not stop.is_set():
                    candidates = []
                    for object_id, (spec, raw) in list(waiting.items()):
                        # Tiny, real schema canaries prevent days of acquiring
                        # a source whose payload/license adapter yields nothing.
                        candidates.append(((raw.byte_count >= 16_000_000, -spec.priority, raw.byte_count, object_id), spec, raw))
                    candidates.sort(key=lambda value: value[0])
                    for _, spec, raw in candidates:
                        if (
                            sum(work.reader is not None for work in active.values()) >= readers
                            or len(active) >= readers + 4
                        ):
                            break
                        lease = claim(root / "locks" / "prep-objects" / index_generation / f"{spec.object_id}.flock")
                        if lease is None:
                            continue
                        if (
                            _object_marker(root, index_generation, spec.object_id).exists()
                            or (
                                _failure_marker(root, index_generation, spec.object_id).exists()
                                and failure_blocks(
                                    read_receipt(_failure_marker(root, index_generation, spec.object_id)),
                                    limits_sha256,
                                )
                            )
                        ):
                            lease.close()
                            del waiting[spec.object_id]
                            continue
                        output = _normalized_directory(root, config, spec)
                        work = ObjectWork(spec, raw, lease, output)
                        work.reader = pool.submit(_execute, _reblock_job, spec, raw, dict(config))
                        active[spec.object_id] = work
                        del waiting[spec.object_id]
                        last_work = now
                queue = []
                for object_id, work in active.items():
                    if work.failed:
                        continue
                    for chunk_id, path in work.ready.items():
                        if chunk_id in work.pending or chunk_id in work.indexed:
                            continue
                        screened = work.screened.get(chunk_id)
                        if screened is None or (
                            screened["status"] == "ELIGIBLE_PENDING_OBJECT_COMPLETION"
                            and work.normalized is not None
                        ):
                            heapq.heappush(queue, (-work.spec.priority, 0, object_id, chunk_id, "screen", path))
                        elif screened["status"] == "ELIGIBLE":
                            heapq.heappush(queue, (-work.spec.priority, 1, object_id, chunk_id, "index",
                                                  under_root(root, screened["receipt_path"])))
                while queue and len(jobs) < workers:
                    _, _, object_id, chunk_id, kind, path = heapq.heappop(queue)
                    work = active[object_id]
                    function = screen_chunk if kind == "screen" else index_chunk
                    jobs[(object_id, chunk_id)] = (
                        kind, pool.submit(_execute, function, path, work.spec, dict(config)),
                    )
                    work.pending.add(chunk_id)
                    last_work = now
                atomic_json(status_path, {
                    "schema": "metis17.prep-status/v1", "host": host, "pid": os.getpid(),
                    "job_id": os.environ.get("SLURM_JOB_ID"), "code_commit": commit,
                    "generation": generation, "updated_at": utc_now(),
                    "index_generation": index_generation,
                    "status": "draining" if stop.is_set() else ("preparing" if active else "waiting_for_raw"),
                    "workers": workers, "raw_readers": readers,
                    "active_objects": sorted(active), "queued_objects": len(waiting),
                    "active_chunk_jobs": len(jobs), "totals": dict(totals),
                    "counter_scope": "worker_session_attempts",
                    "elapsed_seconds": now - started,
                    "policy_ready": True, "near_deletion_complete": False,
                    "capacity_confirmation": current_capacity["capacity_confirmation"],
                })
                if not active and (stop.is_set() or now - last_work >= idle_seconds):
                    break
                time.sleep(2)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            atomic_json(status_path, {
                "schema": "metis17.prep-status/v1", "status": "failed",
                "host": host, "pid": os.getpid(), "code_commit": commit,
                "generation": generation, "index_generation": index_generation,
                "active_objects": sorted(active), "updated_at": utc_now(), **safe_error(exc),
            })
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for work in active.values():
                work.lease.close()
    value = json.loads(status_path.read_text())
    atomic_json(status_path, {**value, "status": "stopped", "updated_at": utc_now()})


def _record_failure(
    root: Path, config: Mapping[str, Any], work: ObjectWork,
    stage: str, error: BaseException, owner: str,
) -> None:
    work.failed = True
    detail = error.result if isinstance(error, WorkerFailure) else safe_error(error)
    capacity_pending = isinstance(error, CapacityPending) or detail.get("capacity_pending") is True
    result = {
        "generation": config["generation"], "object_id": work.spec.object_id,
        "index_generation": config["index_generation"], "worker_sha256": _IMPLEMENTATION_SHA256,
        "source_id": work.spec.source_id, "stage": stage,
        "status": "capacity_pending" if capacity_pending else "failed",
        "created_at": utc_now(), **detail,
        "limits_sha256": (
            detail.get("limits_sha256") or digest_json(read_receipt(root / "limits.json"))
            if capacity_pending else None
        ),
    }
    write_receipt(_failure_marker(root, config["index_generation"], work.spec.object_id), result)
    append_event(root, f"prep-errors/{owner}", result)
