"""Keep tokenizer collection, training and stable ID caching ahead of acquisition."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping

from .admission import claim
from .cli import _stop_event, code_commit, safe_error
from .common import atomic_json, digest_json, read_receipt, under_root, utc_now, write_receipt
from .tokenizer_pipeline import run_tokenizer_step, tokenize_ready_partition
from .worker import EventTail, worker_configuration


def tokenize_event(
    root: Path, event: Mapping[str, Any], *, generation: str, scratch_dir: Path,
    test_mode: bool = False, working_budget: Any | None = None,
) -> dict[str, Any] | None:
    root = root.resolve()
    if type(test_mode) is not bool:
        raise ValueError("test_mode must be a boolean")
    if event.get("generation") != generation:
        return None
    stage_path = under_root(root, event["receipt_path"])
    stage = read_receipt(stage_path)
    seal = digest_json(stage)
    if seal != event["stage_receipt_sha256"]:
        raise RuntimeError("Tokenizer dispatch event disagrees with its sealed eligibility receipt")
    if (
        stage.get("status") != "ELIGIBLE" or stage.get("eligible") is not True
        or stage.get("training_ready") is not True or stage.get("object_complete") is not True
    ):
        raise RuntimeError("Only EOF-covered, fully eligible text can enter token caching")
    marker = root / "state" / "tokenized-chunks" / generation / seal[:2] / f"{seal}.json"
    if marker.exists():
        previous = read_receipt(marker)
        if previous["stage_receipt_sha256"] != seal:
            raise RuntimeError("Completed token-cache dispatch identity changed")
        return previous
    paths = [under_root(root, value["path"]) for value in stage["chunks"]]
    if len(set(paths)) != len(paths):
        raise RuntimeError("A token-cache partition repeats an input artifact")
    result = (
        tokenize_ready_partition(
            root, paths, scratch_dir=scratch_dir, partition_id=seal, generation=generation,
            stage_receipt_path=stage_path, stage_receipt_sha256=seal,
            test_mode=test_mode, working_budget=working_budget,
        )
        if paths else None
    )
    receipt = {
        "schema": "metis17.tokenized-chunk-dispatch/v1",
        "generation": generation, "stage_receipt_sha256": seal,
        "stage_receipt_path": str(stage_path.relative_to(root)),
        "partition_id": seal, "result": result, "created_at": utc_now(),
    }
    write_receipt(marker, receipt)
    return receipt


def tokenizer_service(
    root: Path, *, scratch_dir: Path, poll_seconds: float = 30,
    maximum_seconds: float = 172_000, test_mode: bool = False,
    working_budget: Any | None = None,
) -> None:
    root = root.resolve()
    if poll_seconds <= 0 or maximum_seconds <= 0:
        raise ValueError("Tokenizer service intervals must be positive")
    if type(test_mode) is not bool:
        raise ValueError("test_mode must be a boolean")
    progress_path = root / "status" / "tokenizer.json"
    identity = {
        "schema": "metis17.tokenizer-service/v1",
        "host": socket.gethostname(), "pid": os.getpid(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "code_commit": code_commit(),
    }
    lock = None
    try:
        lock = claim(root / "locks" / "tokenizer-service.flock")
        if lock is None:
            raise RuntimeError("Another tokenizer service already owns this release")
        atomic_json(progress_path, {**identity, "status": "initializing", "updated_at": utc_now()})
        config, _ = worker_configuration(root)
        generation = config["generation"]
        identity["generation"] = generation
        cursor_path = root / "state" / "tokenizer-dispatch" / f"{generation}.json"
        tail = EventTail()
        tokenized_events = ignored_events = 0
        if cursor_path.exists():
            cursor = read_receipt(cursor_path)
            if cursor["generation"] != generation:
                raise RuntimeError("Tokenizer dispatch checkpoint belongs to another eligibility generation")
            tokenized_events = cursor["processed_eligible_events"]
            ignored_events = cursor["ignored_other_generation_events"]
            tail.positions = {
                under_root(root, entry["path"]): (entry["inode"], entry["offset"])
                for entry in cursor["journals"]
            }
        stop = _stop_event()
        started = time.monotonic()
        summary: dict[str, Any] = {}
        while not stop.is_set() and not (root / "STOP").exists():
            if time.monotonic() - started >= maximum_seconds:
                break
            atomic_json(progress_path, {
                **identity, "status": "advancing_tokenizer", "tokenizer": summary, "updated_at": utc_now(),
            })
            previous_chunks = summary.get("candidate_chunks", 0)
            result = run_tokenizer_step(
                root, scratch_dir=scratch_dir, generation=generation, test_mode=test_mode,
                working_budget=working_budget,
            )
            state = result["status"]
            if state not in {"WAITING", "SAMPLE_READY", "TRAINED", "BLOCKED"}:
                raise RuntimeError(f"Unrecognized tokenizer pipeline state: {state}")
            summary = {key: result[key] for key in (
                "status", "activity", "production", "target_bytes", "required_category_bytes",
                "admitted_rows", "admitted_characters", "inventory_bytes", "sample_attempts",
                "ignored_other_generation_events", "ignored_unscoped_events",
                "available_bytes_are_metadata_bounds_not_training_credits",
                "error", "tokenizer_sha256", "tokenizer_release_sha256",
                "required_source_bytes", "required_language_bytes",
                "admitted_source_characters", "admitted_language_characters",
            ) if key in result}
            summary["candidate_chunks"] = len(result.get("chunks", {}))
            made_progress = summary["candidate_chunks"] > previous_chunks
            atomic_json(progress_path, {
                **identity, "status": state.lower(), "tokenizer": summary,
                "processed_eligible_events": tokenized_events, "updated_at": utc_now(),
            })
            if state == "TRAINED":
                for journal in sorted((root / "events" / "eligible").glob("*.jsonl")):
                    if stop.is_set() or time.monotonic() - started >= maximum_seconds:
                        break
                    events = tail.read(journal, maximum_events=32)
                    if not events:
                        continue
                    made_progress = True
                    for event in events:
                        receipt = tokenize_event(
                            root, event, generation=generation, scratch_dir=scratch_dir, test_mode=test_mode,
                            working_budget=working_budget,
                        )
                        if receipt is not None:
                            tokenized_events += 1
                        else:
                            ignored_events += 1
                        atomic_json(progress_path, {
                            **identity, "status": "tokenizing", "tokenizer": summary,
                            "processed_eligible_events": tokenized_events,
                            "ignored_other_generation_events": ignored_events,
                            "updated_at": utc_now(),
                        })
                    # A cursor advances only after every event in its read has
                    # a durable cache receipt. Interrupted batches replay safely.
                    write_receipt(cursor_path, {
                        "schema": "metis17.tokenizer-dispatch-cursor/v1", "generation": generation,
                        "processed_eligible_events": tokenized_events,
                        "ignored_other_generation_events": ignored_events,
                        "journals": [
                            {"path": str(path.relative_to(root)), "inode": inode, "offset": offset}
                            for path, (inode, offset) in sorted(tail.positions.items())
                        ],
                    })
            if state != "SAMPLE_READY" and not made_progress:
                stop.wait(poll_seconds)
        atomic_json(progress_path, {
            **identity, "status": "stopped", "tokenizer": summary,
            "processed_eligible_events": tokenized_events, "updated_at": utc_now(),
        })
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        if lock is not None:
            atomic_json(progress_path, {**identity, "status": "failed", "updated_at": utc_now(), **safe_error(exc)})
        raise
    finally:
        if lock is not None:
            lock.close()
