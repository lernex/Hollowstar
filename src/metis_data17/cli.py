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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests
import yaml

from .acquisition import CapacityPending, DownloadFailure, download_object, file_lock, receipt_path
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


def download_service(root: Path, *, workers: int | None = None) -> None:
    run = load_run(root)
    settings = run["config"]["download"]
    host = socket.gethostname().split(".", 1)[0]
    if host not in settings["hosts"]:
        raise RuntimeError("This host is not an approved acquisition endpoint")
    origins = set(settings["hosts"][host])
    stop = _stop_event()
    committed: set[str] = set()
    preview_counts: dict[str, int] = {}
    completed_bytes = 0
    for event in read_events(root / "events" / "raw" / f"{host}.jsonl"):
        if event["object_id"] not in committed:
            committed.add(event["object_id"])
            completed_bytes += int(event["byte_count"])
            group = str(event["admission_group"])
            preview_counts[group] = preview_counts.get(group, 0) + 1
    specs: dict[str, ObjectSpec] = {}
    loaded_pages: set[Path] = set()
    pending: dict[str, list[tuple[int, int, str]]] = {}
    active: dict[Future[RawReceipt], ObjectSpec] = {}
    blocked_sources: set[str] = set()
    retry_at: dict[str, float] = {}
    paused_capacity = False
    limit_stamp = 0
    status_path = root / "status" / f"download-{host}.json"
    baseline = _interface_counters()
    commit = code_commit()
    budget_path = root / "state" / "intake-budget.json"
    resumable = set(read_receipt(budget_path)["inflight"]) if budget_path.exists() else set()
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
                            if spec.object_id in specs:
                                raise RuntimeError("Duplicate object identity in active acquisition")
                            specs[spec.object_id] = spec
                            if spec.object_id not in committed:
                                group = str(spec.policy.get("admission_group", spec.source_id))
                                heapq.heappush(pending.setdefault(group, []), download_order_key(spec, resumable))
                        loaded_pages.add(page)
                limits = _limits(root)
                new_stamp = (root / "limits.json").stat().st_mtime_ns
                if new_stamp != limit_stamp:
                    paused_capacity = False
                    limit_stamp = new_stamp
                completed = [future for future in active if future.done()]
                for future in completed:
                    spec = active.pop(future)
                    group = str(spec.policy.get("admission_group", spec.source_id))
                    try:
                        receipt = future.result()
                    except CapacityPending:
                        paused_capacity = True
                        heapq.heappush(pending.setdefault(group, []), download_order_key(spec, resumable))
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
                            completed_bytes += receipt.byte_count
                            preview_counts[group] = preview_counts.get(group, 0) + 1
                if not stop.is_set() and not paused_capacity:
                    while len(active) < (workers or int(settings["workers_per_host"])):
                        choices: list[tuple[tuple[int, int, str], str]] = []
                        for group, heap in pending.items():
                            while heap and heap[0][2] in committed:
                                heapq.heappop(heap)
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
                            inflight_group = sum(
                                str(item.policy.get("admission_group", item.source_id)) == group
                                for item in active.values()
                            )
                            if not admitted and preview_counts.get(group, 0) + inflight_group >= int(settings["admission_objects_per_group"]):
                                continue
                            if retry_at.get(candidate.object_id, 0) > time.monotonic():
                                continue
                            choices.append((heap[0], group))
                        if not choices:
                            break
                        key, group = min(choices)
                        heapq.heappop(pending[group])
                        spec = specs[key[2]]
                        future = pool.submit(
                            download_object, spec, root, limits,
                            attempts=int(settings["attempts"]),
                            timeout=float(settings["timeout_seconds"]),
                        )
                        active[future] = spec
                atomic_json(status_path, {
                    "schema": "metis17.download-status/v1",
                    "host": host,
                    "pid": os.getpid(),
                    "code_commit": commit,
                    "updated_at": utc_now(),
                    "status": "downloading" if active else ("capacity_pending" if paused_capacity else "waiting_for_catalogue_or_admission"),
                    "intake_paused_for_capacity": paused_capacity,
                    "worker_limit": workers or int(settings["workers_per_host"]),
                    "active_objects": [item.object_id for item in active.values()],
                    "catalogued_objects": len(specs),
                    "completed_objects": len(committed),
                    "completed_payload_bytes": completed_bytes,
                    "blocked_sources": sorted(blocked_sources),
                    "approved_origins": sorted(origins),
                    "interface": "ens2f3",
                    "interface_baseline": baseline,
                    "interface_now": _interface_counters(),
                    "capacity_confirmation": limits["capacity_confirmation"],
                    "max_raw_bytes": limits["max_raw_bytes"],
                })
                if not active and stop.is_set():
                    break
                time.sleep(min(2.0, float(settings["poll_seconds"])))


def status(root: Path) -> dict[str, Any]:
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
        "policy_ready": (root / "policy" / "CURRENT.json").exists(),
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
    download = sub.add_parser("download")
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--workers", type=int)
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
    elif args.command == "download":
        download_service(root, workers=args.workers)
    elif args.command == "import-policy":
        from .policy import import_policy

        print(canonical_json(import_policy(root, args.source_directory, registry_path=args.registry)))
    elif args.command == "status":
        print(json.dumps(status(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
