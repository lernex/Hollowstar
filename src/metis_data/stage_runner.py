from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import numpy as np
from tokenizers import Tokenizer

from .config import load_profile, repository_root
from .download import run_download_task
from .manifest import load_manifest
from .quality import evaluate_quality, priority_score
from .state import StateStore, utc_now
from .tokenizer import train_tokenizer, validate_tokenizer
from .selection import build_selection, hamilton_apportion
from .download import sha256_file


def _paths(profile: dict[str, Any]) -> tuple[Path, StateStore]:
    root = Path(profile["storage"]["lustre_root"])
    state = StateStore(root / profile["storage"]["directories"]["state"])
    return root, state


def _require_safety_space(profile: dict[str, Any], stage: str) -> None:
    root = Path(profile["storage"]["lustre_root"])
    safety_bytes = int(float(profile["storage"].get("safety_free_tb", 0)) * 1_000_000_000_000)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < safety_bytes:
        raise RuntimeError(
            f"Refusing stage {stage}: {free_bytes:,} free bytes is below the configured "
            f"{safety_bytes:,}-byte safety reserve"
        )


def _manifest(profile: dict[str, Any]) -> dict[str, Any]:
    path = Path(profile["manifest"])
    if not path.is_absolute():
        path = repository_root() / path
    return load_manifest(path)


def _safe_metadata(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"text", "content", "code"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
        elif isinstance(value, dict):
            output[key] = {str(k): v for k, v in value.items() if isinstance(v, (str, int, float, bool)) or v is None}
        elif isinstance(value, list) and len(value) <= 100:
            output[key] = [v for v in value if isinstance(v, (str, int, float, bool)) or v is None]
    nested = output.pop("metadata", None)
    if isinstance(nested, dict):
        # Dataset cards frequently place the only quality/provenance evidence in a
        # nested metadata struct.  Preserve the struct for provenance and expose
        # non-conflicting scalar fields to the fail-closed quality gate.
        output["upstream_metadata"] = nested
        for key, value in nested.items():
            output.setdefault(str(key), value)
    return output


def _first_scalar(metadata: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = metadata.get(name)
        if isinstance(value, (str, int, float, bool)) and value != "":
            return value
    return None


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _normalization_evidence(
    row: dict[str, Any],
    source: dict[str, Any],
    file_record: dict[str, Any],
) -> dict[str, Any]:
    """Map upstream evidence to canonical fields without fabricating scores.

    A source-level reviewed/accepted license is valid evidence for every row.
    Per-record sources still require an actual row license and therefore fail
    closed when it is absent.
    """

    metadata = _safe_metadata(row)
    aliases = {
        "quality_score": ("quality_score", "score", "quality_rating"),
        "educational_score": ("educational_score", "education_score", "edu_score", "score"),
        "math_score": ("math_score", "score", "quality_score"),
        "ocr_confidence": ("ocr_confidence", "ocr_score", "text_extraction_score"),
        "language_probability": ("language_probability", "language_score", "language_confidence", "lang_score"),
        "capture_date": ("capture_date", "date", "crawl_date", "timestamp"),
        "publication_date": ("publication_date", "published", "published_at", "date"),
        "canonical_url": ("canonical_url", "url", "source_url"),
        "version": ("version", "documentation_version", "release", "tag"),
        "license": ("license", "license_name", "repo_license", "spdx_license"),
    }
    for canonical, names in aliases.items():
        value = _first_scalar(metadata, names)
        if value is not None:
            if canonical.endswith("_score") or canonical.endswith("_confidence") or canonical == "language_probability":
                numeric = _numeric_score(value)
                if numeric is not None:
                    metadata[canonical] = numeric
            else:
                metadata[canonical] = value
    if not metadata.get("license"):
        licenses = metadata.get("licenses")
        if isinstance(licenses, list):
            values = sorted({str(value).strip() for value in licenses if str(value).strip()})
            if values:
                metadata["license"] = ",".join(values)

    quality = str(metadata.get("quality", "")).lower()
    source_file = str(file_record.get("repo_path", "")).lower()
    partition_quality = quality or source_file
    if "medium-high-quality" in partition_quality:
        metadata["quality_score"] = max(float(metadata.get("quality_score", 0.0)), 0.80)
    elif "high-quality" in partition_quality or "/4plus/" in f"/{source_file}":
        metadata["quality_score"] = max(float(metadata.get("quality_score", 0.0)), 0.90)
    if source["id"] == "fineweb_edu":
        # The released corpus already applies the documented score >= 3 gate.
        metadata.setdefault("educational_score", 3.0)
    if source["id"] == "nemotron_cc_math_4plus":
        metadata.setdefault("math_score", 4.0)
    elif source["id"] == "nemotron_cc_math_unique_3":
        metadata.setdefault("math_score", 3.0)

    language = str(_first_scalar(metadata, ("language", "lang", "language_code")) or "").lower()
    if language in {"en", "eng", "english", "en-latn", "eng_latn"}:
        metadata.setdefault("language_probability", 1.0)
    profile_name = source["processing"]["quality_profile"]
    if profile_name != "multilingual_native_v1" and (
        source["category"] == "code"
        or "translated_english" in profile_name
        or "eng_latn" in str(file_record.get("repo_path", "")).lower()
    ):
        # Code has no natural-language label, translated-English partitions are
        # source-filtered, and FinePDFs' eng_Latn path is an upstream language
        # partition.  General web still requires row-level detector evidence.
        metadata.setdefault("language_probability", 1.0)

    license_status = source["license"]["status"]
    if license_status in {"reviewed", "requires_acceptance", "requires_review", "public_domain_or_reviewed"}:
        metadata.setdefault("license", source["license"]["expression"])
    return metadata


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    name = path.name.lower()
    if name.endswith(".parquet"):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            yield from batch.to_pylist()
        return
    if name.endswith(".jsonl.zst"):
        import zstandard as zstd

        raw = path.open("rb")
        handle = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
    elif name.endswith(".jsonl.gz") or name.endswith(".json.gz"):
        handle = gzip.open(path, "rt", encoding="utf-8")
    elif name.endswith(".jsonl") or name.endswith(".json"):
        handle = path.open("r", encoding="utf-8")
    else:
        return
    with handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _text_from_row(row: dict[str, Any]) -> str:
    for key in ("text", "content", "code", "body", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _normalize_task(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    import zstandard as zstd

    root, state = _paths(profile)
    lock = state.read("sources.lock.json")
    task = lock["download_tasks"][task_index]
    completion = state.read("completed", "download", f"{task['task_id']}.json")
    if not completion:
        raise RuntimeError(f"Download prerequisite is incomplete: {task['task_id']}")
    manifest = _manifest(profile)
    sources = {source["id"]: source for source in manifest["sources"]}
    output_dir = root / profile["storage"]["directories"]["normalized"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"task-{task_index:06d}.jsonl.zst"
    report = output_dir / f"task-{task_index:06d}.report.json"
    if report.exists() and output.exists():
        return json.loads(report.read_text(encoding="utf-8"))
    counts = {"input": 0, "accepted": 0, "rejected": 0, "no_text": 0, "remote_plans": 0}
    rejection_reasons: dict[str, int] = {}
    with output.open("wb") as raw:
        with zstd.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for file_record in completion["files"]:
                    if file_record.get("kind") == "remote_source_plan":
                        counts["remote_plans"] += 1
                        continue
                    source_id = file_record["source_id"]
                    source = sources[source_id]
                    profile_name = source["processing"]["quality_profile"]
                    source_priority = int(source["processing"].get("priority", 1))
                    for row_index, row in enumerate(_iter_rows(Path(file_record["local_path"]))):
                        counts["input"] += 1
                        text = _text_from_row(row)
                        if not text:
                            counts["no_text"] += 1
                            continue
                        metadata = _normalization_evidence(row, source, file_record)
                        metadata.update(
                            {
                                "source_id": source_id,
                                "category": source["category"],
                                "source_revision": file_record.get("revision"),
                                "source_file": file_record.get("repo_path"),
                                "license_status": source["license"]["status"],
                                "generated": bool(source["provenance"].get("generated")),
                                "human_original": not bool(source["provenance"].get("generated"))
                                and not bool(source["provenance"].get("transformed")),
                                "fresh": bool(source["provenance"].get("fresh")),
                            }
                        )
                        if source["license"]["status"] in {"per_record_required", "inherited", "requires_review"} and not metadata.get("license"):
                            counts["rejected"] += 1
                            rejection_reasons["missing_license"] = rejection_reasons.get("missing_license", 0) + 1
                            continue
                        decision = evaluate_quality(
                            text,
                            profile_name=profile_name,
                            metadata=metadata,
                            fail_closed=bool(profile.get("gates", {}).get("fail_closed", True)),
                        )
                        if not decision.keep:
                            counts["rejected"] += 1
                            rejection_reasons[decision.reason] = rejection_reasons.get(decision.reason, 0) + 1
                            continue
                        doc_id = row.get("id") or row.get("uuid") or f"{source_id}:{task_index}:{row_index}"
                        payload = {
                            "id": str(doc_id),
                            "text": text,
                            "metadata": {
                                **metadata,
                                "priority": priority_score(source_priority, metadata),
                                "quality_features": decision.features,
                            },
                        }
                        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                        counts["accepted"] += 1
    if counts["remote_plans"]:
        raise RuntimeError(
            f"Normalization task {task_index} contains {counts['remote_plans']} unresolved remote acquisition plan(s). "
            "A selection plan is not training data; materialize it before submitting the build graph."
        )
    if counts["accepted"] == 0:
        raise RuntimeError(f"Normalization task {task_index} accepted zero records; see rejection counts {rejection_reasons}")
    payload = {
        "stage": "normalize",
        "task_index": task_index,
        "output": str(output),
        "counts": counts,
        "rejection_reasons": rejection_reasons,
        "completed_at": utc_now(),
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.complete("normalize", f"task-{task_index:06d}", payload)
    return payload


def _content(doc: Any) -> str:
    return str(doc.text)


def _priority(doc: Any) -> int:
    return int(doc.metadata.get("priority", 1))


def _local_executor(profile: dict[str, Any], stage: str, task_index: int, tasks: int, pipeline: list[Any]) -> None:
    from datatrove.executor.local import LocalPipelineExecutor

    root, _ = _paths(profile)
    logs = root / profile["storage"]["directories"]["logs"] / stage
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=str(logs),
        tasks=tasks,
        workers=1,
        local_tasks=1,
        local_rank_offset=task_index,
        skip_completed=True,
    )
    executor.run()


def _datatrove_stage(profile: dict[str, Any], stage: str, task_index: int) -> dict[str, Any]:
    from datatrove.pipeline.dedup import ExactDedupConfig, ExactDedupFilter, ExactDedupSignature, ExactFindDedups
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupFilter
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    from datatrove.utils.hashing import HashConfig
    from .datatrove_blocks import build_regex_word_tokenizer

    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    normalized = root / directories["normalized"]
    eligible = root / directories["eligible"]
    dedup = root / directories["dedup"]
    contamination = root / directories["contamination"]
    total_tasks = max(1, len(state.read("sources.lock.json")["download_tasks"]))
    finder_tasks = int(profile["scheduler"]["exact_dedup"]["find_tasks"])
    mh_profile = profile["scheduler"]["minhash"]
    mh_config = MinhashConfig(
        n_grams=int(mh_profile["n_grams"]),
        num_buckets=int(mh_profile["num_buckets"]),
        hashes_per_bucket=int(mh_profile["hashes_per_bucket"]),
        seed=16062026,
        hash_config=HashConfig(precision=64),
    )
    exact_config = ExactDedupConfig(
        content_getter=_content,
        document_priority=_priority,
        hash_config=HashConfig(precision=64),
        only_dedup_in_index=False,
    )
    reader = JsonlReader(str(normalized), glob_pattern="task-*.jsonl.zst", compression="zstd", shuffle_files=False)
    exact_sig = dedup / "exact" / "signatures"
    exact_dups = dedup / "exact" / "duplicates"
    exact_output = eligible / "exact"
    mh_sig = dedup / "minhash" / "signatures"
    mh_buckets = dedup / "minhash" / "buckets"
    mh_remove = dedup / "minhash" / "remove_ids"
    mh_output = eligible / "near-deduped"

    if stage == "exact_signature":
        _local_executor(profile, stage, task_index, total_tasks, [reader, ExactDedupSignature(str(exact_sig), exact_config, finder_workers=finder_tasks)])
    elif stage == "exact_find":
        _local_executor(profile, stage, task_index, finder_tasks, [ExactFindDedups(str(exact_sig), str(exact_dups), exact_config)])
    elif stage == "exact_filter":
        _local_executor(profile, stage, task_index, total_tasks, [reader, ExactDedupFilter(str(exact_dups), exact_config), JsonlWriter(str(exact_output))])
    elif stage == "minhash_signature":
        exact_reader = JsonlReader(str(exact_output), shuffle_files=False)
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [exact_reader, MinhashDedupSignature(str(mh_sig), config=mh_config, language=build_regex_word_tokenizer())],
        )
    elif stage == "minhash_buckets":
        _local_executor(profile, stage, task_index, mh_config.num_buckets, [MinhashDedupBuckets(str(mh_sig), str(mh_buckets), config=mh_config)])
    elif stage == "minhash_cluster":
        from .datatrove_blocks import build_priority_minhash_removals

        cluster_report = build_priority_minhash_removals(
            mh_buckets,
            mh_remove,
            exact_output,
            total_tasks=total_tasks,
        )
        state.write("minhash-cluster-report.json", payload=cluster_report)
    elif stage == "minhash_filter":
        exact_reader = JsonlReader(str(exact_output), shuffle_files=False)
        _local_executor(profile, stage, task_index, total_tasks, [exact_reader, MinhashDedupFilter(str(mh_remove)), JsonlWriter(str(mh_output))])
    elif stage == "decontam_index":
        holdouts = contamination / "holdouts.jsonl"
        if not holdouts.exists():
            raise RuntimeError(f"Fail-closed: benchmark holdout bundle is missing at {holdouts}")
        from .datatrove_blocks import save_contamination_index
        from .decontaminate import ContaminationIndex

        texts = []
        for row in _iter_rows(holdouts):
            texts.append(str(row.get("text", "")))
            metadata = row.get("metadata", {})
            if metadata.get("query"):
                texts.append(str(metadata["query"]))
        index = ContaminationIndex.build(texts, ngram_size=13, minimum_matching_ngrams=2)
        save_contamination_index(index, contamination / "index.json")
    elif stage == "decontam_filter":
        from .datatrove_blocks import build_datatrove_decontamination_filter

        index_path = contamination / "index.json"
        if not index_path.exists():
            raise RuntimeError(f"Fail-closed: decontamination index is missing at {index_path}")
        mh_reader = JsonlReader(str(mh_output), shuffle_files=False)
        _local_executor(
            profile,
            stage,
            task_index,
            total_tasks,
            [mh_reader, build_datatrove_decontamination_filter(index_path), JsonlWriter(str(eligible / "decontaminated"))],
        )
    else:
        raise ValueError(f"Unsupported DataTrove stage {stage}")
    payload = {"stage": stage, "task_index": task_index, "completed_at": utc_now()}
    state.complete(stage, f"task-{task_index:06d}", payload)
    return payload


def _iter_jsonl_folder(folder: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(folder.glob("**/*.jsonl*")):
        yield from _iter_rows(path)


def _tokenizer_sample(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    manifest = _manifest(profile)
    eligible = root / profile["storage"]["directories"]["eligible"] / "decontaminated"
    output_dir = root / profile["storage"]["directories"]["tokenizer"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "sample.jsonl"
    target = int(manifest["tokenizer"]["sample_target_bytes"])
    minimum_category = int(manifest["tokenizer"]["min_sample_bytes_per_category"])
    category_weights = {
        category["id"]: sum(int(value) for value in category["phase_tokens"].values())
        for category in manifest["categories"]
    }
    base = minimum_category * len(category_weights)
    if base > target:
        raise RuntimeError("Tokenizer category floors exceed the total sample target")
    category_extra = hamilton_apportion(target - base, category_weights)
    category_targets = {category: minimum_category + category_extra[category] for category in category_weights}
    source_targets: dict[str, int] = {}
    for category, category_target in category_targets.items():
        sources = [source for source in manifest["sources"] if source["category"] == category]
        weights = {source["id"]: sum(int(value) for value in source["phase_tokens"].values()) for source in sources}
        source_targets.update(hamilton_apportion(category_target, weights))
    written: dict[str, int] = {source_id: 0 for source_id in source_targets}
    total = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in _iter_jsonl_folder(eligible):
            metadata = row.get("metadata", {})
            source_id = metadata.get("source_id") or row.get("source_id")
            if source_id not in source_targets or written[source_id] >= source_targets[source_id]:
                continue
            text = str(row.get("text", ""))
            encoded = text.encode("utf-8")
            handle.write(json.dumps({"source_id": source_id, "category": metadata.get("category"), "text": text}, ensure_ascii=False) + "\n")
            written[source_id] += len(encoded)
            total += len(encoded)
    short = {source_id: source_targets[source_id] - actual for source_id, actual in written.items() if actual < source_targets[source_id]}
    if short:
        raise RuntimeError(f"Tokenizer sample exhausted before its stratified source targets: {short}")
    category_written = {
        category: sum(written[source["id"]] for source in manifest["sources"] if source["category"] == category)
        for category in category_targets
    }
    payload = {
        "stage": "tokenizer_sample",
        "bytes": total,
        "target_bytes": target,
        "category_targets": category_targets,
        "category_bytes": category_written,
        "source_targets": source_targets,
        "source_bytes": written,
        "output": str(output),
    }
    state.complete("tokenizer_sample", "task-000000", payload)
    return payload


def _tokenizer_train(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    manifest = _manifest(profile)
    output_dir = root / profile["storage"]["directories"]["tokenizer"]
    sample = output_dir / "sample.jsonl"

    def texts() -> Iterator[str]:
        for row in _iter_rows(sample):
            yield str(row["text"])

    payload = train_tokenizer(
        texts(),
        output_dir=output_dir,
        vocabulary_size=int(manifest["tokenizer"]["vocabulary_size_including_special_tokens"]),
        special_tokens=list(manifest["tokenizer"]["special_tokens"]),
    )
    audit_limits: dict[str, int] = {}

    def audit_samples() -> Iterator[dict[str, Any]]:
        for row in _iter_rows(sample):
            category = str(row.get("category", "unknown"))
            if audit_limits.get(category, 0) >= 500:
                continue
            audit_limits[category] = audit_limits.get(category, 0) + 1
            yield row

    validation = validate_tokenizer(output_dir / "tokenizer.json", audit_samples())
    validation["sampled_documents_by_category"] = audit_limits
    validation_path = output_dir / "TOKENIZER_VALIDATION.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not validation["ok"]:
        raise RuntimeError(
            f"Tokenizer round-trip validation failed for {validation['roundtrip_failure_count']} sampled documents"
        )
    payload["validation"] = validation
    payload["validation_path"] = str(validation_path)
    state.complete("tokenizer_train", "task-000000", payload)
    return payload


def _token_count(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    import zstandard as zstd

    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    source_dir = root / directories["eligible"] / "decontaminated"
    output_dir = root / directories["token_counts"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"task-{task_index:06d}.jsonl.zst"
    report_path = output_dir / f"task-{task_index:06d}.report.json"
    if report_path.exists() and output.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    total_tasks = max(1, len(state.read("sources.lock.json")["download_tasks"]))
    paths = sorted(source_dir.glob("**/*.jsonl*"))
    assigned = paths[task_index::total_tasks]
    tokenizer_path = root / directories["tokenizer"] / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise RuntimeError("Tokenizer is missing <|endoftext|>")
    source_tokens: dict[str, int] = {}
    documents = 0
    with output.open("wb") as raw:
        with zstd.ZstdCompressor(level=6).stream_writer(raw) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for path in assigned:
                    for row in _iter_rows(path):
                        metadata = row.get("metadata", {})
                        text = str(row.get("text", ""))
                        source_id = str(metadata.get("source_id", row.get("source_id", "")))
                        if not source_id or not text:
                            continue
                        token_count = len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
                        doc_id = str(row.get("id", metadata.get("doc_id", f"{task_index}:{documents}")))
                        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        payload = {
                            "source_id": source_id,
                            "category": metadata.get("category"),
                            "doc_id": doc_id,
                            "text": text,
                            "token_count": token_count,
                            "content_sha256": content_sha,
                            "generated": bool(metadata.get("generated", False)),
                            "priority": int(metadata.get("priority", 1)),
                            "license": metadata.get("license"),
                            "license_status": metadata.get("license_status"),
                        }
                        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                        source_tokens[source_id] = source_tokens.get(source_id, 0) + token_count
                        documents += 1
    payload = {
        "stage": "token_count",
        "task_index": task_index,
        "documents": documents,
        "tokens": sum(source_tokens.values()),
        "source_tokens": source_tokens,
        "output": str(output),
        "completed_at": utc_now(),
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    state.complete("token_count", f"task-{task_index:06d}", payload)
    return payload


def _select(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    output_root = root / directories["selected"]
    selection_path = output_root / "SELECTION.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = _manifest(profile)
    reports = sorted((root / directories["token_counts"]).glob("task-*.report.json"))
    expected_reports = max(1, len(state.read("sources.lock.json")["download_tasks"]))
    if len(reports) != expected_reports:
        raise RuntimeError(f"Expected {expected_reports} token-count reports, found {len(reports)}")
    eligible_tokens: dict[str, int] = {}
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for source_id, tokens in report["source_tokens"].items():
            eligible_tokens[source_id] = eligible_tokens.get(source_id, 0) + int(tokens)

    def records() -> Iterator[dict[str, Any]]:
        for path in sorted((root / directories["token_counts"]).glob("task-*.jsonl.zst")):
            yield from _iter_rows(path)

    payload = build_selection(
        records(),
        manifest=manifest,
        eligible_tokens=eligible_tokens,
        output_root=output_root,
        shard_tokens=int(profile["storage"]["final_shard_tokens"]),
    )
    state.complete("select", "task-000000", payload)
    return payload


def _pack(profile: dict[str, Any], task_index: int) -> dict[str, Any]:
    root, state = _paths(profile)
    directories = profile["storage"]["directories"]
    selected = root / directories["selected"]
    selection = json.loads((selected / "SELECTION.json").read_text(encoding="utf-8"))
    matching = [shard for shard in selection["shards"] if int(shard["global_index"]) == task_index]
    if len(matching) != 1:
        raise ValueError(f"Selection does not contain exactly one global shard {task_index}")
    shard = matching[0]
    task_id = f"task-{task_index:06d}"
    if state.is_complete("pack", task_id):
        return state.read("completed", "pack", f"{task_id}.json")
    tokenizer = Tokenizer.from_file(str(root / directories["tokenizer"] / "tokenizer.json"))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise RuntimeError("Tokenizer is missing EOS")
    release_root = root / directories["release"]
    phase_dir = release_root / shard["phase"].replace("_", "-")
    phase_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard-{int(shard['phase_index']):05d}"
    binary = phase_dir / f"{stem}.bin"
    index = phase_dir / f"{stem}.index.jsonl"
    temporary_binary = binary.with_suffix(".bin.incomplete")
    temporary_index = index.with_suffix(".jsonl.incomplete")
    written = 0
    documents = 0
    source_tokens: dict[str, int] = {}
    license_tokens: dict[str, dict[str, int]] = {}
    generated_tokens = 0
    missing_license_tokens = 0
    with temporary_binary.open("wb") as binary_handle, temporary_index.open("w", encoding="utf-8") as index_handle:
        for record in _iter_rows(Path(shard["path"])):
            ids = tokenizer.encode(str(record["text"]), add_special_tokens=False).ids + [eos_id]
            start = int(record["token_start"])
            count = int(record["token_count"])
            selected_ids = ids[start : start + count]
            if len(selected_ids) != count:
                raise RuntimeError(f"Token slice is out of bounds for {record['source_id']}:{record['doc_id']}")
            array = np.asarray(selected_ids, dtype=np.uint16)
            binary_handle.write(array.tobytes())
            index_handle.write(
                json.dumps(
                    {
                        "start": written,
                        "end": written + count,
                        "source_id": record["source_id"],
                        "doc_id": record["doc_id"],
                        "replay": bool(record["replay"]),
                        "content_sha256": record.get("content_sha256"),
                        "license": record.get("license"),
                        "license_status": record.get("license_status"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            written += count
            documents += 1
            source_tokens[record["source_id"]] = source_tokens.get(record["source_id"], 0) + count
            license_expression = str(record.get("license") or "")
            if not license_expression:
                missing_license_tokens += count
            else:
                by_license = license_tokens.setdefault(record["source_id"], {})
                by_license[license_expression] = by_license.get(license_expression, 0) + count
            if record.get("generated"):
                generated_tokens += count
    if written != int(shard["target_tokens"]):
        raise RuntimeError(f"Pack task {task_index} wrote {written:,}, expected {int(shard['target_tokens']):,}")
    if shard["phase"] == "phase_c" and generated_tokens:
        raise RuntimeError(f"Phase C shard contains {generated_tokens:,} generated tokens")
    temporary_binary.replace(binary)
    temporary_index.replace(index)
    payload = {
        "stage": "pack",
        "task_index": task_index,
        "phase": shard["phase"],
        "phase_index": shard["phase_index"],
        "tokens": written,
        "documents": documents,
        "source_tokens": source_tokens,
        "license_tokens": license_tokens,
        "missing_license_tokens": missing_license_tokens,
        "generated_tokens": generated_tokens,
        "binary": str(binary),
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": sha256_file(binary),
        "index": str(index),
        "index_sha256": sha256_file(index),
        "completed_at": utc_now(),
    }
    state.complete("pack", task_id, payload)
    return payload


def _verify(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    manifest = _manifest(profile)
    if profile.get("gates", {}).get("require_license_ledger") and not profile.get("gates", {}).get("license_review_complete", False):
        raise RuntimeError("Fail-closed: the source/license review has not been marked complete in the Portage profile")
    directories = profile["storage"]["directories"]
    selection = json.loads((root / directories["selected"] / "SELECTION.json").read_text(encoding="utf-8"))
    pack_reports = []
    for shard in selection["shards"]:
        task_id = f"task-{int(shard['global_index']):06d}"
        report = state.read("completed", "pack", f"{task_id}.json")
        if not report:
            raise RuntimeError(f"Pack completion is missing: {task_id}")
        binary = Path(report["binary"])
        if binary.stat().st_size != int(report["tokens"]) * 2:
            raise RuntimeError(f"uint16 byte size mismatch: {binary}")
        if sha256_file(binary) != report["binary_sha256"]:
            raise RuntimeError(f"Binary checksum mismatch: {binary}")
        index_path = Path(report["index"])
        if sha256_file(index_path) != report["index_sha256"]:
            raise RuntimeError(f"Index checksum mismatch: {index_path}")
        if report["phase"] == "phase_c" and int(report["generated_tokens"]):
            raise RuntimeError(f"Generated data found in phase C: {binary}")
        if int(report.get("missing_license_tokens", 0)):
            raise RuntimeError(f"Shard contains records without license evidence: {binary}")
        pack_reports.append(report)
    actual: dict[str, dict[str, int]] = {
        source["id"]: {phase: 0 for phase in ("phase_a", "phase_b", "phase_c")}
        for source in manifest["sources"]
    }
    phase_tokens = {phase: 0 for phase in ("phase_a", "phase_b", "phase_c")}
    for report in pack_reports:
        phase = report["phase"]
        phase_tokens[phase] += int(report["tokens"])
        for source_id, tokens in report["source_tokens"].items():
            actual[source_id][phase] += int(tokens)
    expected_phase = {phase: int(manifest["schedule"]["phases"][phase]["target_tokens"]) for phase in phase_tokens}
    if phase_tokens != expected_phase:
        raise RuntimeError(f"Phase totals mismatch: {phase_tokens} != {expected_phase}")
    mismatches = {}
    for source in manifest["sources"]:
        expected = {phase: int(source["phase_tokens"].get(phase, 0)) for phase in phase_tokens}
        if actual[source["id"]] != expected:
            mismatches[source["id"]] = {"actual": actual[source["id"]], "expected": expected}
    if mismatches:
        raise RuntimeError(f"Source/phase token mismatches: {mismatches}")
    release_root = root / directories["release"]
    provenance_root = release_root / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)
    ledger_path = provenance_root / "LICENSE_LEDGER.jsonl"
    with ledger_path.open("w", encoding="utf-8") as ledger:
        for source in manifest["sources"]:
            license_payload = source["license"]
            observed: dict[str, int] = {}
            for report in pack_reports:
                for expression, tokens in report.get("license_tokens", {}).get(source["id"], {}).items():
                    observed[expression] = observed.get(expression, 0) + int(tokens)
            ledger.write(
                json.dumps(
                    {
                        "source_id": source["id"],
                        "license_status": license_payload["status"],
                        "license_expression": license_payload["expression"],
                        "observed_license_tokens": observed,
                        "training_recipe_disposition": "verified_for_training",
                        "data_publication_requires_separate_review": True,
                        "access": source["access"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    shard_manifest_path = provenance_root / "SHARDS.jsonl"
    with shard_manifest_path.open("w", encoding="utf-8") as shard_manifest:
        for report in sorted(pack_reports, key=lambda item: int(item["task_index"])):
            shard_manifest.write(
                json.dumps(
                    {
                        "task_index": int(report["task_index"]),
                        "phase": report["phase"],
                        "tokens": int(report["tokens"]),
                        "binary": str(Path(report["binary"]).relative_to(release_root)),
                        "binary_sha256": report["binary_sha256"],
                        "index": str(Path(report["index"]).relative_to(release_root)),
                        "index_sha256": report["index_sha256"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    payload = {
        "schema": "metis.verification/v1",
        "ok": True,
        "verified_at": utc_now(),
        "target_tokens": sum(phase_tokens.values()),
        "phase_tokens": phase_tokens,
        "source_phase_tokens": actual,
        "shards": len(pack_reports),
        "license_ledger": str(ledger_path),
        "license_ledger_sha256": sha256_file(ledger_path),
        "shard_manifest": str(shard_manifest_path),
        "shard_manifest_sha256": sha256_file(shard_manifest_path),
    }
    state.write("VERIFICATION.json", payload=payload)
    state.complete("verify", "task-000000", payload)
    return payload


def _release(profile: dict[str, Any]) -> dict[str, Any]:
    root, state = _paths(profile)
    manifest = _manifest(profile)
    directories = profile["storage"]["directories"]
    verification = state.read("VERIFICATION.json")
    if not verification or not verification.get("ok"):
        raise RuntimeError("Verified release gate has not passed")
    if profile.get("gates", {}).get("require_license_ledger"):
        ledger = root / directories["release"] / "provenance" / "LICENSE_LEDGER.jsonl"
        if not ledger.exists():
            raise RuntimeError(f"Fail-closed: license ledger is missing at {ledger}")
    release_root = root / directories["release"]
    tokenizer = root / directories["tokenizer"] / "tokenizer.json"
    selection = root / directories["selected"] / "SELECTION.json"
    source_lock = state.path("sources.lock.json")
    release_root.mkdir(parents=True, exist_ok=True)
    release_tokenizer = release_root / "tokenizer"
    release_manifests = release_root / "manifests"
    release_reports = release_root / "reports"
    release_tokenizer.mkdir(parents=True, exist_ok=True)
    release_manifests.mkdir(parents=True, exist_ok=True)
    release_reports.mkdir(parents=True, exist_ok=True)
    tokenizer_root = tokenizer.parent
    for name in ("tokenizer.json", "vocab.json", "TOKENIZER_RELEASE.json", "TOKENIZER_VALIDATION.json"):
        source = tokenizer_root / name
        if not source.exists():
            raise RuntimeError(f"Tokenizer release artifact is missing: {source}")
        shutil.copy2(source, release_tokenizer / name)
    shutil.copy2(Path(manifest["_path"]), release_manifests / "metis-1.6.yaml")
    manifest_repository = repository_root() / "manifests"
    for subdirectory in ("sources", "contamination", "registries", "licenses"):
        source_directory = manifest_repository / subdirectory
        if source_directory.exists():
            shutil.copytree(source_directory, release_manifests / subdirectory, dirs_exist_ok=True)
    shutil.copy2(source_lock, release_manifests / "sources.lock.json")
    shutil.copy2(selection, release_manifests / "SELECTION.json")
    verification_path = release_reports / "VERIFICATION.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "schema": "metis.data-release/v1",
        "release": manifest["release"],
        "released_at": utc_now(),
        "target_tokens": verification["target_tokens"],
        "phase_tokens": verification["phase_tokens"],
        "token_dtype": profile["storage"]["final_token_dtype"],
        "tokenizer_sha256": sha256_file(tokenizer),
        "selection_sha256": sha256_file(selection),
        "source_lock_sha256": sha256_file(source_lock),
        "manifest_sha256": sha256_file(Path(manifest["_path"])),
        "manifest_bundle_sha256": _tree_sha256(release_manifests),
        "verification": verification,
        "artifacts": {
            "tokenizer": "tokenizer/tokenizer.json",
            "source_lock": "manifests/sources.lock.json",
            "selection": "manifests/SELECTION.json",
            "verification": "reports/VERIFICATION.json",
            "license_ledger": "provenance/LICENSE_LEDGER.jsonl",
            "shard_manifest": "provenance/SHARDS.jsonl",
        },
    }
    from .state import atomic_json

    atomic_json(release_root / "RELEASE.json", payload)
    state.complete("release", "task-000000", payload)
    return payload


def run_stage(profile: dict[str, Any], stage: str, task_index: int) -> dict[str, Any]:
    _require_safety_space(profile, stage)
    if stage == "download":
        return run_download_task(profile, task_index)
    if stage == "normalize":
        return _normalize_task(profile, task_index)
    if stage in {
        "exact_signature", "exact_find", "exact_filter", "minhash_signature", "minhash_buckets",
        "minhash_cluster", "minhash_filter", "decontam_index", "decontam_filter",
    }:
        return _datatrove_stage(profile, stage, task_index)
    if stage == "tokenizer_sample":
        return _tokenizer_sample(profile)
    if stage == "tokenizer_train":
        return _tokenizer_train(profile)
    if stage == "token_count":
        return _token_count(profile, task_index)
    if stage == "select":
        return _select(profile)
    if stage == "pack":
        return _pack(profile, task_index)
    if stage == "verify":
        return _verify(profile)
    if stage == "release":
        return _release(profile)
    raise RuntimeError(f"Unknown stage {stage!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args(argv)
    _, profile = load_profile(args.profile)
    try:
        payload = run_stage(profile, args.stage, args.task_index)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
