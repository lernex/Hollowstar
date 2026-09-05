from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests
import yaml

from .acquisition import CapacityPending, DownloadFailure, DownloadPaused, download_object, file_lock, receipt_path
from .catalogue import resolve_source
from .common import ObjectSpec, RawReceipt, atomic_json, canonical_json, digest_json, read_receipt, utc_now, write_receipt


def code_root() -> Path:
    return Path(__file__).resolve().parents[2]


def code_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(code_root()), "rev-parse", "HEAD"], text=True
    ).strip()


def _stop_event() -> threading.Event:
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _sig, _frame: stop.set())
    return stop


def _limits(root: Path) -> dict[str, Any]:
    value = read_receipt(root / "limits.json")
    confirmation = value.get("capacity_confirmation")
    if confirmation not in {"pending", "administrator-confirmed", "unlimited"}:
        raise ValueError("Unsupported capacity confirmation")
    for name in ("max_raw_bytes", "max_working_bytes"):
        if type(value.get(name)) is not int or value[name] <= 0:
            raise ValueError(f"Invalid capacity bound: {name}")
    if confirmation == "pending" and (
        value["max_raw_bytes"] > 400_000_000_000
        or value["max_working_bytes"] > 2_000_000_000_000
    ):
        raise CapacityPending("Full acquisition needs explicit storage-capacity confirmation")
    if value["max_raw_bytes"] > 200_000_000_000_000:
        raise ValueError("Raw allocation exceeds the 200 TB envelope")
    return value


def init_run(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if root.name != "metis-1.7" and not root.name.startswith("metis-1.7-"):
        raise ValueError("Use a dedicated metis-1.7 release directory, not an existing data root")
    if root == code_root() or root.is_relative_to(code_root()):
        raise ValueError("The code checkout cannot be a data root")
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema") != "metis17.pipeline/v1":
        raise ValueError("Unsupported pipeline configuration")
    if config["tokenizer"]["vocabulary_size"] != 131072 or config["tokenizer"]["split_digits"] is not True:
        raise ValueError("Metis-1.7 requires 131072 vocabulary entries and digit splitting")
    if config["schedule"] != {
        "tst_source_tokens": 30_000_000_000_000,
        "tst_bag_size": 16,
        "ntp_source_tokens": 5_000_000_000_000,
    }:
        raise ValueError("The 30T/16 + 5T training brief changed")
    ids = [source["id"] for source in config["sources"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source identity in activation")
    if any(source["kind"] not in {"hf", "hplt", "cc"} for source in config["sources"]):
        raise ValueError("Repository/pointer reconstruction is not admitted")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / "RUN.json"
    if path.exists():
        existing = read_receipt(path)
        if existing["config_sha256"] != digest_json(config):
            raise RuntimeError("Existing run configuration differs; do not mutate live source definitions")
        _limits(root)
        return existing
    payload = {
        "schema": "metis17.run/v1",
        "release": config["release"],
        "created_at": utc_now(),
        "initial_code_commit": code_commit(),
        "root": str(root),
        "config": config,
        "config_sha256": digest_json(config),
        "full_capacity_approved": config["limits"]["capacity_confirmation"] != "pending",
    }
    write_receipt(root / "limits.json", config["limits"])
    _limits(root)
    write_receipt(path, payload)
    return payload


def load_run(root: Path) -> dict[str, Any]:
    value = read_receipt(root / "RUN.json")
    if value["config_sha256"] != digest_json(value["config"]):
        raise ValueError("Run configuration digest mismatch")
    return value


def append_event(root: Path, stream: str, value: dict[str, Any]) -> None:
    path = root / "events" / f"{stream}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**value, "recorded_at": utc_now()}
    payload["event_sha256"] = digest_json(payload)
    data = (canonical_json(payload) + "\n").encode("utf-8")
    maximum = 1_048_576
    if len(data) > maximum:
        raise ValueError("Event must contain bounded metadata, not corpus text")
    with file_lock(path.with_name(path.name + ".lock")):
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            size = os.lseek(descriptor, 0, os.SEEK_END)
            if size:
                os.lseek(descriptor, max(0, size - maximum), os.SEEK_SET)
                tail = os.read(descriptor, maximum)
                if not tail.endswith(b"\n"):
                    last = tail.rfind(b"\n")
                    if last < 0 and size > maximum:
                        raise RuntimeError("Event journal has an oversized uncommitted tail")
                    os.ftruncate(descriptor, max(0, size - maximum) + last + 1)
            written = os.write(descriptor, data)
            if written != len(data):
                raise OSError("Incomplete event-journal append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def safe_error(exc: BaseException) -> dict[str, str]:
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    detail = re.sub(r"https?://[^\s'\"<>]+", lambda m: m[0].split("?", 1)[0], detail)
    detail = re.sub(r"\b(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{16,}", "<redacted>", detail)
    return {"error_type": type(exc).__name__, "traceback": detail[-16000:]}


def read_events(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                # A final interrupted append is not a committed event.
                break
            event = json.loads(raw)
            expected = event.pop("event_sha256", None)
            if expected != digest_json(event):
                raise RuntimeError(f"Event journal is corrupt: {path}")
            yield event


def _interface_counters() -> dict[str, int | None]:
    base = Path("/sys/class/net/ens2f3")
    if not base.is_dir():
        return {"rx_bytes": None, "tx_bytes": None, "speed_mbps": None}
    return {
        "rx_bytes": int((base / "statistics/rx_bytes").read_text()),
        "tx_bytes": int((base / "statistics/tx_bytes").read_text()),
        "speed_mbps": int((base / "speed").read_text()),
    }


def download_order_key(spec: ObjectSpec, resumable: set[str]) -> tuple[int, int, str]:
    lane = 0 if spec.object_id in resumable else (1 if spec.policy.get("bootstrap") else 2)
    return lane, -spec.priority, spec.object_id


def select_download_group(
    choices: list[tuple[tuple[int, int, str], str]],
    specs: dict[str, ObjectSpec],
    active: Iterable[ObjectSpec],
) -> tuple[tuple[int, int, str], str]:
    active_origins = Counter(urlsplit(spec.url).hostname for spec in active)
    # One connection to the fast origin is not enough when the other origin
    # throttles an entire pool. Preserve priority within each independent origin.
    return min(choices, key=lambda item: (
        active_origins[urlsplit(specs[item[0][2]].url).hostname], item,
    ))


def download_owner(
    spec: ObjectSpec, hosts: dict[str, list[str]], shared_origins: set[str],
    resumable_owners: dict[str, str],
) -> str:
    origin = urlsplit(spec.url).hostname
    eligible = sorted(name for name, origins in hosts.items()
                      if origin in origins or origin in shared_origins)
    if not eligible:
        raise ValueError("The object has no approved acquisition host")
    previous = resumable_owners.get(spec.object_id)
    if previous is not None:
        previous = previous.split(".", 1)[0]
        if previous not in eligible:
            raise RuntimeError("A partial download belongs to an unapproved host")
        return previous
    if spec.policy.get("bootstrap"):
        eligible = sorted(name for name, origins in hosts.items() if origin in origins)
    return eligible[int(spec.object_id[:16], 16) % len(eligible)]


def intake_candidate_fits(
    spec: ObjectSpec, intake: dict[str, Any], limits: dict[str, Any],
) -> bool:
    inflight = intake["inflight"]
    others = sum(value["bytes"] for key, value in inflight.items() if key != spec.object_id)
    available = limits["max_raw_bytes"] - intake["raw_bytes"] - others
    estimate = spec.expected_bytes
    if estimate is None:
        estimate = inflight.get(spec.object_id, {}).get(
            "bytes", limits.get("max_unknown_object_bytes", 32_000_000_000),
        )
    # This conservative scheduling hint never grants storage. The downloader
    # still reserves atomically after learning the actual response length.
    return estimate <= available


def resolve_command(root: Path, *, source_ids: set[str] | None = None) -> dict[str, Any]:
    run = load_run(root)
    results: list[dict[str, Any]] = []
    for source in run["config"]["sources"]:
        if source_ids is not None and source["id"] not in source_ids:
            continue
        try:
            result = resolve_source(root, source)
        except (requests.RequestException, RuntimeError, ValueError, OSError) as exc:
            append_event(root, "catalogue", {
                "source_id": source["id"],
                "status": "failed",
                **safe_error(exc),
            })
            results.append({"source_id": source["id"], "status": "failed", "error_type": type(exc).__name__})
            continue
        entry = {
            "source_id": source["id"],
            "status": "complete",
            "objects": result["objects"],
            "known_bytes": result["known_bytes"],
            "unknown_size_objects": result["unknown_size_objects"],
        }
        append_event(root, "catalogue", entry)
        results.append(entry)
        print(canonical_json(entry), flush=True)
    return {"sources": results, "ok": bool(results) and all(row["status"] == "complete" for row in results)}


def activate_batch(
    root: Path, config_path: Path, *, reconcile_incomplete: bool = False,
) -> dict[str, Any]:
    load_run(root)
    batch = yaml.safe_load(config_path.read_text())
    if batch.get("schema") != "metis17.activation/v1" or not isinstance(batch.get("sources"), list):
        raise ValueError("A sealed content-source activation batch is required")
    identifiers = [source["id"] for source in batch["sources"]]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Activation source identities must be nonempty and unique")
    if any(source.get("kind") not in {"hf", "hplt", "cc"} for source in batch["sources"]):
        raise ValueError("Only packaged content and sequential archive objects may be activated")
    batch_id = digest_json(batch)
    path = root / "activations" / batch_id / "BATCH.json"
    if path.exists() and read_receipt(path) != batch:
        raise RuntimeError("Immutable activation batch changed")
    write_receipt(path, batch)
    results = []
    for source in batch["sources"]:
        try:
            result = (
                resolve_source(root, source, reconcile_incomplete=True)
                if reconcile_incomplete else resolve_source(root, source)
            )
        except (requests.RequestException, RuntimeError, ValueError, OSError) as exc:
            row = {"source_id": source["id"], "status": "failed", **safe_error(exc)}
            append_event(root, "activation-errors", row)
        else:
            row = {"source_id": source["id"], "status": "complete",
                   "objects": result["objects"], "known_bytes": result["known_bytes"]}
        results.append(row)
        print(canonical_json(row), flush=True)
    return {"batch_id": batch_id, "sources": results, "ok": all(row["status"] == "complete" for row in results)}


def download_service(
    root: Path, *, workers: int | None = None, shared_origins: set[str] | None = None,
) -> None:
    run = load_run(root)
    settings = run["config"]["download"]
    host = socket.gethostname().split(".", 1)[0]
    if host not in settings["hosts"]:
        raise RuntimeError("This host is not an approved acquisition endpoint")
    shared_origins = set(shared_origins or ())
    approved = {origin for values in settings["hosts"].values() for origin in values}
    if not shared_origins <= approved:
        raise ValueError("Shared origins must already be approved by the frozen run")
    origins = set(settings["hosts"][host]) | shared_origins
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("A positive download worker count is required")
    stop = _stop_event()
    committed: set[str] = set()
    preview_counts: dict[str, int] = {}
    completed_bytes = 0
    published: set[str] = set()
    for journal in sorted((root / "events" / "raw").glob("*.jsonl")):
        for event in read_events(journal):
            if event["object_id"] not in published:
                published.add(event["object_id"])
                group = str(event["admission_group"])
                preview_counts[group] = preview_counts.get(group, 0) + 1
            if journal.stem == host and event["object_id"] not in committed:
                committed.add(event["object_id"])
                completed_bytes += int(event["byte_count"])
    specs: dict[str, ObjectSpec] = {}
    loaded_pages: set[Path] = set()
    pending: dict[str, list[tuple[int, int, str]]] = {}
    active: dict[Future[RawReceipt], ObjectSpec] = {}
    blocked_sources: set[str] = set()
    retry_at: dict[str, float] = {}
    capacity_blocked: dict[str, ObjectSpec] = {}
    limit_stamp = 0
    budget_stamp = 0
    status_path = root / "status" / f"download-{host}.json"
    baseline = _interface_counters()
    commit = code_commit()
    budget_path = root / "state" / "intake-budget.json"
    reservations = read_receipt(budget_path)["inflight"] if budget_path.exists() else {}
    resumable = set(reservations)
    resumable_owners = {key: value["host"] for key, value in reservations.items()}
    with file_lock(root / "locks" / f"download-daemon-{host}.lock", timeout=2):
        with ThreadPoolExecutor(max_workers=workers or int(settings["workers_per_host"])) as pool:
            while not stop.is_set() or active:
                if (root / "STOP").exists():
                    stop.set()
                for descriptor_path in sorted((root / "catalogue" / "active").glob("*.json")):
                    descriptor = read_receipt(descriptor_path)
                    directory = (root / descriptor["directory"]).resolve()
                    if not directory.is_relative_to((root / "catalogue").resolve()):
                        raise RuntimeError("Active catalogue path escaped root")
                    for page in sorted(directory.glob("page-*.json")):
                        if page in loaded_pages:
                            continue
                        value = read_receipt(page)
                        if value["source_hash"] != descriptor["source_hash"]:
                            raise RuntimeError("Source catalogue identity changed")
                        for record in value["objects"]:
                            spec = ObjectSpec.from_dict(record)
                            if urlsplit(spec.url).hostname not in origins:
                                continue
                            if download_owner(spec, settings["hosts"], shared_origins, resumable_owners) != host:
                                continue
                            if spec.object_id in specs:
                                raise RuntimeError("Duplicate object identity in active acquisition")
                            specs[spec.object_id] = spec
                            if spec.object_id not in published:
                                group = str(spec.policy.get("admission_group", spec.source_id))
                                heapq.heappush(pending.setdefault(group, []), download_order_key(spec, resumable))
                        loaded_pages.add(page)
                limits = _limits(root)
                intake = read_receipt(budget_path) if budget_path.exists() else {"raw_bytes": 0, "inflight": {}}
                new_stamp = (root / "limits.json").stat().st_mtime_ns
                new_budget_stamp = budget_path.stat().st_mtime_ns if budget_path.exists() else 0
                if new_stamp != limit_stamp or new_budget_stamp != budget_stamp:
                    for spec in capacity_blocked.values():
                        group = str(spec.policy.get("admission_group", spec.source_id))
                        heapq.heappush(pending.setdefault(group, []), download_order_key(spec, resumable))
                    capacity_blocked.clear()
                    limit_stamp, budget_stamp = new_stamp, new_budget_stamp
                completed = [future for future in active if future.done()]
                for future in completed:
                    spec = active.pop(future)
                    group = str(spec.policy.get("admission_group", spec.source_id))
                    try:
                        receipt = future.result()
                    except DownloadPaused:
                        if not stop.is_set():
                            raise RuntimeError("A download paused without a requested shutdown")
                    except CapacityPending:
                        # A large object or exhausted source must not pause
                        # smaller, higher-value work from independent sources.
                        capacity_blocked[spec.object_id] = spec
                    except PermissionError:
                        blocked_sources.add(spec.source_id)
                        append_event(root, f"download-errors/{host}", {
                            "object_id": spec.object_id, "source_id": spec.source_id, "reason": "access_denied",
                        })
                    except (DownloadFailure, TimeoutError, OSError) as exc:
                        retry_at[spec.object_id] = time.monotonic() + 120
                        resumable.add(spec.object_id)
                        heapq.heappush(pending.setdefault(group, []), download_order_key(spec, resumable))
                        append_event(root, f"download-errors/{host}", {
                            "object_id": spec.object_id, "source_id": spec.source_id, **safe_error(exc),
                        })
                    else:
                        if spec.object_id not in committed:
                            append_event(root, f"raw/{host}", {
                                **receipt.to_dict(),
                                "admission_group": group,
                                "spec": spec.to_dict(),
                            })
                            committed.add(spec.object_id)
                            published.add(spec.object_id)
                            completed_bytes += receipt.byte_count
                            preview_counts[group] = preview_counts.get(group, 0) + 1
                if not stop.is_set():
                    while len(active) < (workers or int(settings["workers_per_host"])):
                        choices: list[tuple[tuple[int, int, str], str]] = []
                        deferred: dict[str, list[tuple[int, int, str]]] = {}
                        for group, heap in pending.items():
                            while heap:
                                candidate = specs[heap[0][2]]
                                if candidate.object_id in committed:
                                    heapq.heappop(heap)
                                elif not intake_candidate_fits(candidate, intake, limits):
                                    heapq.heappop(heap)
                                    capacity_blocked[candidate.object_id] = candidate
                                elif retry_at.get(candidate.object_id, 0) > time.monotonic():
                                    deferred.setdefault(group, []).append(heapq.heappop(heap))
                                else:
                                    break
                            if not heap:
                                continue
                            candidate = specs[heap[0][2]]
                            if candidate.source_id in blocked_sources:
                                continue
                            admitted_path = root / "admissions" / f"{digest_json(group)}.json"
                            admitted = False
                            if admitted_path.exists():
                                admission = read_receipt(admitted_path)
                                if admission["admission_group"] != group:
                                    raise RuntimeError("Source admission identity mismatch")
                                admitted = admission["status"] == "admitted"
                            if not admitted:
                                native_hosts = sorted(
                                    name for name, allowed in settings["hosts"].items()
                                    if urlsplit(candidate.url).hostname in allowed
                                )
                                if host != native_hosts[0]:
                                    continue
                            inflight_group = sum(
                                str(item.policy.get("admission_group", item.source_id)) == group
                                for item in active.values()
                            )
                            if not admitted and preview_counts.get(group, 0) + inflight_group >= int(settings["admission_objects_per_group"]):
                                continue
                            choices.append((heap[0], group))
                        selected = select_download_group(choices, specs, active.values()) if choices else None
                        if selected is not None:
                            heapq.heappop(pending[selected[1]])
                        for group, keys in deferred.items():
                            for key in keys:
                                heapq.heappush(pending[group], key)
                        if selected is None:
                            break
                        key, group = selected
                        spec = specs[key[2]]
                        future = pool.submit(
                            download_object, spec, root, limits,
                            attempts=int(settings["attempts"]),
                            timeout=float(settings["timeout_seconds"]),
                            stop_event=stop,
                        )
                        active[future] = spec
                atomic_json(status_path, {
                    "schema": "metis17.download-status/v1",
                    "host": host,
                    "pid": os.getpid(),
                    "code_commit": commit,
                    "updated_at": utc_now(),
                    "status": (
                        ("draining" if active else "stopped") if stop.is_set()
                        else "downloading" if active
                        else "capacity_pending" if capacity_blocked
                        else "waiting_for_catalogue_or_admission"
                    ),
                    "intake_paused_for_capacity": bool(capacity_blocked) and not active,
                    "capacity_blocked_objects": len(capacity_blocked),
                    "worker_limit": workers or int(settings["workers_per_host"]),
                    "active_objects": [item.object_id for item in active.values()],
                    "catalogued_objects": len(specs),
                    "completed_objects": len(committed),
                    "completed_payload_bytes": completed_bytes,
                    "blocked_sources": sorted(blocked_sources),
                    "approved_origins": sorted(origins),
                    "shared_origins": sorted(shared_origins),
                    "routing_policy": "object-hash-with-existing-partial-owner/v1",
                    "interface": "ens2f3",
                    "interface_baseline": baseline,
                    "interface_now": _interface_counters(),
                    "capacity_confirmation": limits["capacity_confirmation"],
                    "max_raw_bytes": limits["max_raw_bytes"],
                    "intake_reserved_bytes": sum(value["bytes"] for value in intake["inflight"].values()),
                    "intake_headroom_bytes": (
                        limits["max_raw_bytes"] - intake["raw_bytes"]
                        - sum(value["bytes"] for value in intake["inflight"].values())
                    ),
                })
                if not active and stop.is_set():
                    break
                time.sleep(min(2.0, float(settings["poll_seconds"])))


def status(root: Path) -> dict[str, Any]:
    from .policy import policy_config
    from .storage import WorkingBudget

    run = load_run(root)
    downloaders = {
        path.stem.removeprefix("download-"): json.loads(path.read_text())
        for path in (root / "status").glob("download-*.json")
    }
    prep = {
        path.stem: json.loads(path.read_text())
        for path in (root / "status").glob("prep-*.json")
    }
    intake = root / "state" / "intake-budget.json"
    return {
        "release": run["release"],
        "root": str(root),
        "limits": _limits(root),
        "downloaders": downloaders,
        "prep_workers": prep,
        "intake": read_receipt(intake) if intake.exists() else None,
        "working_storage": (
            WorkingBudget(root).snapshot()
            if (root / "state" / "working-budget" / "total.json").exists() else None
        ),
        "policy_ready": policy_config(root)["policy_ready"],
        "tokenizer_ready": (root / "tokenizer" / "TOKENIZER_RELEASE.json").exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metis-data17")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--config", type=Path, default=code_root() / "configs/metis17/pipeline.yaml")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--root", type=Path, required=True)
    resolve.add_argument("--source", action="append")
    activate = sub.add_parser("activate")
    activate.add_argument("--root", type=Path, required=True)
    activate.add_argument("--config", type=Path, required=True)
    activate.add_argument("--reconcile-incomplete", action="store_true")
    download = sub.add_parser("download")
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--workers", type=int)
    download.add_argument("--share-origin", action="append", default=[])
    prepare = sub.add_parser("prep")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--workers", type=int, default=32)
    prepare.add_argument("--raw-readers", type=int)
    prepare.add_argument("--idle-seconds", type=float, default=600)
    prepare.add_argument("--maximum-seconds", type=float, default=42000)
    supervise = sub.add_parser("supervise-prep")
    supervise.add_argument("--root", type=Path, required=True)
    supervise.add_argument("--python", type=Path, required=True)
    supervise.add_argument("--maximum-nodes", type=int, default=4)
    supervise.add_argument("--workers", type=int, default=32)
    policy = sub.add_parser("import-policy")
    policy.add_argument("--root", type=Path, required=True)
    policy.add_argument("--source-directory", type=Path, required=True)
    policy.add_argument("--registry", type=Path)
    report = sub.add_parser("status")
    report.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.command == "init":
        result = init_run(root, args.config)
        print(canonical_json({key: value for key, value in result.items() if key != "config"}))
    elif args.command == "resolve":
        result = resolve_command(root, source_ids=set(args.source) if args.source else None)
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    elif args.command == "activate":
        result = activate_batch(root, args.config, reconcile_incomplete=args.reconcile_incomplete)
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    elif args.command == "download":
        download_service(root, workers=args.workers, shared_origins=set(args.share_origin))
    elif args.command == "prep":
        from .worker import prep_service

        prep_service(
            root, workers=args.workers, raw_readers=args.raw_readers,
            idle_seconds=args.idle_seconds, maximum_seconds=args.maximum_seconds,
        )
    elif args.command == "supervise-prep":
        from .runtime import supervise_prep

        supervise_prep(
            root, code_root(), args.python.expanduser().absolute(),
            maximum_nodes=args.maximum_nodes, workers_per_node=args.workers,
        )
    elif args.command == "import-policy":
        from .policy import import_policy

        print(canonical_json(import_policy(root, args.source_directory, registry_path=args.registry)))
    elif args.command == "status":
        print(json.dumps(status(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
