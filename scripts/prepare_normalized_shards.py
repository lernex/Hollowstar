from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import zstandard as zstd

from data_mixture import (
    SoftwareHeritageFetcher,
    SourceSpec,
    build_planned_counts,
    clean_text,
    extract_doc_id,
    extract_text,
    iter_text_chunks,
    load_source_specs_from_config,
    load_source_dataset,
    passes_quality_filters,
    passes_row_filters,
)
from s3_artifacts import S3ArtifactStore, join_s3_uri


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def purge_local_hf_cache() -> None:
    """Trim local HF dataset cache to bound disk usage on small prep boxes."""
    roots = []
    hf_home = os.environ.get("HF_HOME")
    hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    if hf_datasets_cache:
        roots.append(Path(hf_datasets_cache))

    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.name == "downloads":
                # Hugging Face keeps active/incomplete download bookkeeping here.
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def normalize_source(
    *,
    spec_payload: dict[str, Any],
    target_docs: int,
    output_dir: str,
    shard_docs: int,
    zstd_level: int,
    progress_interval: int,
    s3_prefix: str | None,
    shard_name_prefix: str = "shard",
    manifest_name: str = "manifest.json",
    clear_stale: bool = True,
) -> dict[str, Any]:
    spec = SourceSpec.from_dict(spec_payload)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / manifest_name

    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = None
        if (
            existing_manifest
            and int(existing_manifest.get("docs_written", 0)) >= target_docs
            and existing_manifest.get("shards")
        ):
            print(
                f"[{spec.name}] Reusing completed normalized source: "
                f"{existing_manifest.get('docs_written', 0)}/{target_docs} docs.",
                flush=True,
            )
            return existing_manifest

    # If a previous run died mid-source, remove stale shard files before
    # rewriting shard-00001 so old extra shards cannot linger in local manifests.
    if clear_stale:
        for stale_path in output_path.glob(f"{shard_name_prefix}-*.jsonl.zst"):
            stale_path.unlink(missing_ok=True)

    if target_docs <= 0:
        manifest = {
            "source": spec.name,
            "target_docs": target_docs,
            "docs_written": 0,
            "attempted_rows": 0,
            "skipped_examples": 0,
            "fetch_errors": 0,
            "source_counts": {},
            "shards": [],
        }
        write_json(manifest_path, manifest)
        return manifest

    dataset = load_source_dataset(spec)
    iterator = iter(dataset)
    fetcher = SoftwareHeritageFetcher()
    store = S3ArtifactStore() if s3_prefix else None
    partition_label = ""
    filters = spec.filters or {}
    if int(filters.get("file_partition_count") or filters.get("dataset_shard_count") or 1) > 1:
        partition_index = int(filters.get("file_partition_index") or filters.get("dataset_shard_index") or 0)
        partition_count = int(filters.get("file_partition_count") or filters.get("dataset_shard_count") or 1)
        partition_label = f"/p{partition_index:02d}of{partition_count:02d}"

    total_docs = 0
    attempted_rows = 0
    skipped_examples = 0
    fetch_errors = 0
    source_counts: Counter[str] = Counter()
    shards: list[dict[str, Any]] = []
    shard_sources: Counter[str] = Counter()
    shard_count = 0
    shard_index = 0
    shard_path: Path | None = None
    raw_handle = None
    zstd_handle = None
    text_handle: io.TextIOWrapper | None = None
    chunk_long_documents = bool((spec.filters or {}).get("chunk_long_documents"))
    chunk_target_chars = int((spec.filters or {}).get("chunk_target_chars") or max(spec.min_chars, 4096))
    chunk_max_chars = int((spec.filters or {}).get("chunk_max_chars") or max(chunk_target_chars * 2, chunk_target_chars + 512))

    def open_next_shard() -> None:
        nonlocal shard_index, shard_count, shard_path, raw_handle, zstd_handle, text_handle, shard_sources
        shard_index += 1
        shard_count = 0
        shard_sources = Counter()
        shard_path = output_path / f"{shard_name_prefix}-{shard_index:05d}.jsonl.zst"
        raw_handle = shard_path.open("wb")
        zstd_handle = zstd.ZstdCompressor(level=zstd_level).stream_writer(raw_handle)
        text_handle = io.TextIOWrapper(zstd_handle, encoding="utf-8")

    def close_current_shard() -> None:
        nonlocal raw_handle, zstd_handle, text_handle
        if shard_path is None or text_handle is None or zstd_handle is None or raw_handle is None:
            return
        if shard_count == 0:
            text_handle.close()
            zstd_handle.close()
            raw_handle.close()
            shard_path.unlink(missing_ok=True)
            raw_handle = None
            zstd_handle = None
            text_handle = None
            return

        text_handle.flush()
        text_handle.close()
        zstd_handle.close()
        raw_handle.close()
        shard_record = {
            "index": shard_index,
            "path": shard_path.name,
            "docs": shard_count,
            "size_bytes": shard_path.stat().st_size,
            "source_counts": dict(shard_sources),
        }
        if s3_prefix:
            shard_record["s3_uri"] = join_s3_uri(s3_prefix, shard_path.name)
            assert store is not None
            store.upload_file(
                local_path=shard_path,
                s3_uri=shard_record["s3_uri"],
                content_type="application/zstd",
            )
        shards.append(shard_record)
        raw_handle = None
        zstd_handle = None
        text_handle = None

    open_next_shard()
    while total_docs < target_docs:
        try:
            row = next(iterator)
        except StopIteration:
            break

        attempted_rows += 1
        if not passes_row_filters(row, spec):
            skipped_examples += 1
            continue

        try:
            text = extract_text(row, spec, fetcher)
        except Exception:
            fetch_errors += 1
            continue

        if not text:
            skipped_examples += 1
            continue

        stripped = clean_text(text, spec)
        if len(stripped) < spec.min_chars:
            skipped_examples += 1
            continue

        if spec.max_chars is not None:
            stripped = stripped[: spec.max_chars].strip()
            if not stripped:
                skipped_examples += 1
                continue

        if not chunk_long_documents and not passes_quality_filters(stripped, spec):
            skipped_examples += 1
            continue

        chunks = [stripped]
        if chunk_long_documents:
            chunks = list(
                iter_text_chunks(
                    stripped,
                    min_chars=spec.min_chars,
                    target_chars=chunk_target_chars,
                    max_chars=chunk_max_chars,
                )
            )

        base_doc_id = extract_doc_id(row, spec, attempted_rows)
        for chunk_index, chunk in enumerate(chunks, start=1):
            if total_docs >= target_docs:
                break
            chunk = chunk.strip()
            if len(chunk) < spec.min_chars or not passes_quality_filters(chunk, spec):
                skipped_examples += 1
                continue
            if text_handle is None:
                open_next_shard()

            payload = {
                "source": spec.name,
                "doc_id": f"{base_doc_id}:{chunk_index}" if chunk_long_documents else base_doc_id,
                "text": chunk,
            }
            text_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total_docs += 1
            shard_count += 1
            source_counts[spec.name] += 1
            shard_sources[spec.name] += 1

            if total_docs == 1 or total_docs % progress_interval == 0:
                print(
                    f"[{spec.name}{partition_label}] Normalized docs written: {total_docs}/{target_docs} | "
                    f"attempted={attempted_rows} skipped={skipped_examples} fetch_errors={fetch_errors}",
                    flush=True,
                )

            if shard_count >= shard_docs:
                close_current_shard()
                open_next_shard()

    close_current_shard()

    manifest = {
        "source": spec.name,
        "dataset_name": spec.dataset_name,
        "dataset_config": spec.dataset_config,
        "split": spec.split,
        "streaming": spec.streaming,
        "target_docs": target_docs,
        "docs_written": total_docs,
        "attempted_rows": attempted_rows,
        "skipped_examples": skipped_examples,
        "fetch_errors": fetch_errors,
        "source_counts": dict(source_counts),
        "shards": shards,
        "output_dir": str(output_path),
        "s3_prefix": s3_prefix,
        "manifest_name": manifest_name,
        "shard_name_prefix": shard_name_prefix,
    }
    write_json(manifest_path, manifest)
    if s3_prefix:
        assert store is not None
        store.upload_text(
            text=json.dumps(manifest, indent=2) + "\n",
            s3_uri=join_s3_uri(s3_prefix, "manifest.json"),
        )

    if total_docs <= 0 and not bool((spec.filters or {}).get("_allow_empty_partition")):
        raise RuntimeError(
            f"Normalized shard build produced zero documents for source {spec.name}. "
            "Check local dataset availability, filters, or upstream availability."
        )

    return manifest


def split_target_docs(total_docs: int, parts: int, part_index: int) -> int:
    base = total_docs // parts
    remainder = total_docs % parts
    return base + (1 if part_index < remainder else 0)


def completed_source_manifest(source_dir: Path, target_docs: int) -> dict[str, Any] | None:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if int(manifest.get("docs_written", 0)) >= target_docs and manifest.get("shards"):
        return manifest
    return None


def clear_source_outputs(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in source_dir.glob("shard*.jsonl.zst"):
        stale_path.unlink(missing_ok=True)
    for stale_path in source_dir.glob("manifest*.json"):
        stale_path.unlink(missing_ok=True)


def clear_partition_outputs(source_dir: Path, part_index: int) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in source_dir.glob(f"shard-p{part_index:03d}*.jsonl.zst"):
        stale_path.unlink(missing_ok=True)
    (source_dir / f"manifest-p{part_index:03d}.json").unlink(missing_ok=True)


def completed_manifest_path(manifest_path: Path, target_docs: int) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if int(manifest.get("docs_written", 0)) >= target_docs and manifest.get("shards"):
        return manifest
    return None


def partition_count_for_target(
    *,
    target_docs: int,
    max_partitions: int,
    partition_target_docs: int,
    partition_min_docs: int,
) -> int:
    if max_partitions <= 1 or target_docs < partition_min_docs:
        return 1
    wanted = max(1, math.ceil(target_docs / max(1, partition_target_docs)))
    return max(1, min(max_partitions, wanted, target_docs))


def partitioned_spec_payload(spec: SourceSpec, *, part_index: int, part_count: int) -> dict[str, Any]:
    payload = spec.to_dict()
    filters = dict(payload.get("filters") or {})
    filters.update(
        {
            "file_partition_index": part_index,
            "file_partition_count": part_count,
            "dataset_shard_index": part_index,
            "dataset_shard_count": part_count,
            "_allow_empty_partition": True,
        }
    )
    payload["filters"] = filters
    return payload


def combine_partition_manifests(
    *,
    spec: SourceSpec,
    target_docs: int,
    output_dir: Path,
    s3_prefix: str | None,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered_manifests = sorted(
        manifests,
        key=lambda item: (
            int(item.get("partition_index", 0)),
            str(item.get("manifest_name", "")),
        ),
    )
    source_counts: Counter[str] = Counter()
    shards: list[dict[str, Any]] = []
    for manifest in ordered_manifests:
        source_counts.update(manifest.get("source_counts") or {})
        for shard in manifest.get("shards") or []:
            shards.append(dict(shard))

    combined = {
        "source": spec.name,
        "dataset_name": spec.dataset_name,
        "dataset_config": spec.dataset_config,
        "split": spec.split,
        "streaming": spec.streaming,
        "target_docs": target_docs,
        "docs_written": sum(int(item.get("docs_written", 0)) for item in ordered_manifests),
        "attempted_rows": sum(int(item.get("attempted_rows", 0)) for item in ordered_manifests),
        "skipped_examples": sum(int(item.get("skipped_examples", 0)) for item in ordered_manifests),
        "fetch_errors": sum(int(item.get("fetch_errors", 0)) for item in ordered_manifests),
        "source_counts": dict(source_counts),
        "shards": shards,
        "partitioned": True,
        "partitions": len(ordered_manifests),
        "partition_manifests": [
            item.get("manifest_name", f"manifest-p{index:03d}.json")
            for index, item in enumerate(ordered_manifests)
        ],
        "output_dir": str(output_dir),
        "s3_prefix": s3_prefix,
    }
    write_json(output_dir / "manifest.json", combined)
    if s3_prefix:
        S3ArtifactStore().upload_text(
            text=json.dumps(combined, indent=2) + "\n",
            s3_uri=join_s3_uri(s3_prefix, "manifest.json"),
        )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reusable normalized Metis text shards from local HF-cached datasets and optionally mirror them to S3."
    )
    parser.add_argument("--mixture-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-docs", type=int, required=True)
    parser.add_argument("--shard-docs", type=int, default=50000)
    parser.add_argument("--zstd-level", type=int, default=6)
    parser.add_argument("--progress-interval", type=int, default=2000)
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--source-partitions",
        type=int,
        default=1,
        help="Maximum file/dataset partitions per large source. Use with high --workers for shard-level parallelism.",
    )
    parser.add_argument(
        "--partition-target-docs",
        type=int,
        default=250000,
        help="Aim for roughly this many target docs per source partition.",
    )
    parser.add_argument(
        "--partition-min-docs",
        type=int,
        default=300000,
        help="Do not partition sources with fewer target docs than this.",
    )
    parser.add_argument(
        "--purge-hf-cache-between-sources",
        action="store_true",
        help="After each source finishes, delete the local HF dataset cache before starting the next source.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    raw_config = json.loads(Path(args.mixture_config).read_text(encoding="utf-8"))
    source_specs = load_source_specs_from_config(raw_config)
    planned_counts = build_planned_counts(
        [spec.weight for spec in source_specs],
        args.max_docs,
    )

    partition_counts = {
        index: partition_count_for_target(
            target_docs=planned_counts.get(index, 0),
            max_partitions=max(1, args.source_partitions),
            partition_target_docs=max(1, args.partition_target_docs),
            partition_min_docs=max(1, args.partition_min_docs),
        )
        for index, _spec in enumerate(source_specs)
    }
    total_jobs = sum(partition_counts.values())
    max_workers = max(1, min(args.workers, max(1, total_jobs)))
    source_results: dict[str, dict[str, Any]] = {}

    if args.purge_hf_cache_between_sources and max_workers != 1:
        raise ValueError("--purge-hf-cache-between-sources requires --workers 1")

    if max_workers == 1 and max(partition_counts.values(), default=1) == 1:
        for index, spec in enumerate(source_specs):
            target_docs = planned_counts.get(index, 0)
            result = normalize_source(
                spec_payload=spec.to_dict(),
                target_docs=target_docs,
                output_dir=str(output_dir / spec.name),
                shard_docs=args.shard_docs,
                zstd_level=args.zstd_level,
                progress_interval=args.progress_interval,
                s3_prefix=join_s3_uri(args.s3_prefix, spec.name) if args.s3_prefix else None,
            )
            source_results[result["source"]] = result
            if args.purge_hf_cache_between_sources:
                purge_local_hf_cache()
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            partition_results: dict[str, list[dict[str, Any]]] = {}
            for index, spec in enumerate(source_specs):
                target_docs = planned_counts.get(index, 0)
                source_dir = output_dir / spec.name
                existing_manifest = completed_source_manifest(source_dir, target_docs)
                if existing_manifest is not None:
                    print(
                        f"[{spec.name}] Reusing completed normalized source: "
                        f"{existing_manifest.get('docs_written', 0)}/{target_docs} docs.",
                        flush=True,
                    )
                    source_results[spec.name] = existing_manifest
                    continue

                part_count = partition_counts.get(index, 1)
                if part_count == 1:
                    clear_source_outputs(source_dir)
                    future = executor.submit(
                        normalize_source,
                        spec_payload=spec.to_dict(),
                        target_docs=target_docs,
                        output_dir=str(source_dir),
                        shard_docs=args.shard_docs,
                        zstd_level=args.zstd_level,
                        progress_interval=args.progress_interval,
                        s3_prefix=join_s3_uri(args.s3_prefix, spec.name) if args.s3_prefix else None,
                        clear_stale=False,
                    )
                    futures[future] = (spec.name, None, part_count, spec, target_docs)
                    continue

                print(
                    f"[{spec.name}] Partitioning source into {part_count} lanes "
                    f"for {target_docs} target docs.",
                    flush=True,
                )
                partition_results.setdefault(spec.name, [])
                for part_index in range(part_count):
                    part_target_docs = split_target_docs(target_docs, part_count, part_index)
                    part_manifest = completed_manifest_path(
                        source_dir / f"manifest-p{part_index:03d}.json",
                        part_target_docs,
                    )
                    if part_manifest is not None:
                        part_manifest["partition_index"] = part_index
                        part_manifest["partition_count"] = part_count
                        partition_results[spec.name].append(part_manifest)
                        print(
                            f"[{spec.name}/p{part_index:02d}of{part_count:02d}] "
                            f"Reusing completed partition: "
                            f"{part_manifest.get('docs_written', 0)}/{part_target_docs} docs.",
                            flush=True,
                        )
                        continue
                    clear_partition_outputs(source_dir, part_index)
                    future = executor.submit(
                        normalize_source,
                        spec_payload=partitioned_spec_payload(
                            spec,
                            part_index=part_index,
                            part_count=part_count,
                        ),
                        target_docs=part_target_docs,
                        output_dir=str(source_dir),
                        shard_docs=args.shard_docs,
                        zstd_level=args.zstd_level,
                        progress_interval=args.progress_interval,
                        s3_prefix=join_s3_uri(args.s3_prefix, spec.name) if args.s3_prefix else None,
                        shard_name_prefix=f"shard-p{part_index:03d}",
                        manifest_name=f"manifest-p{part_index:03d}.json",
                        clear_stale=False,
                    )
                    futures[future] = (spec.name, part_index, part_count, spec, target_docs)

                if len(partition_results[spec.name]) == part_count:
                    source_results[spec.name] = combine_partition_manifests(
                        spec=spec,
                        target_docs=target_docs,
                        output_dir=output_dir / spec.name,
                        s3_prefix=join_s3_uri(args.s3_prefix, spec.name) if args.s3_prefix else None,
                        manifests=partition_results[spec.name],
                    )

            for future in as_completed(futures):
                result = future.result()
                source_name, part_index, part_count, spec, target_docs = futures[future]
                if part_index is None:
                    source_results[result["source"]] = result
                    continue
                result["partition_index"] = part_index
                result["partition_count"] = part_count
                partition_results.setdefault(source_name, []).append(result)

                if len(partition_results[source_name]) == part_count:
                    source_results[source_name] = combine_partition_manifests(
                        spec=spec,
                        target_docs=target_docs,
                        output_dir=output_dir / source_name,
                        s3_prefix=join_s3_uri(args.s3_prefix, source_name) if args.s3_prefix else None,
                        manifests=partition_results[source_name],
                    )

    ordered_results = [source_results[spec.name] for spec in source_specs if spec.name in source_results]
    total_docs = sum(result["docs_written"] for result in ordered_results)
    total_attempted = sum(result["attempted_rows"] for result in ordered_results)
    total_skipped = sum(result["skipped_examples"] for result in ordered_results)
    total_fetch_errors = sum(result["fetch_errors"] for result in ordered_results)

    manifest = {
        "mixture_config": args.mixture_config,
        "max_docs_requested": args.max_docs,
        "docs_written": total_docs,
        "attempted_rows": total_attempted,
        "skipped_examples": total_skipped,
        "fetch_errors": total_fetch_errors,
        "shard_docs": args.shard_docs,
        "zstd_level": args.zstd_level,
        "workers": max_workers,
        "output_dir": str(output_dir),
        "s3_prefix": args.s3_prefix,
        "planned_counts": {source_specs[index].name: count for index, count in sorted(planned_counts.items())},
        "sources": ordered_results,
    }
    write_json(manifest_path, manifest)
    if args.s3_prefix:
        S3ArtifactStore().upload_text(
            text=json.dumps(manifest, indent=2) + "\n",
            s3_uri=join_s3_uri(args.s3_prefix, "manifest.json"),
        )

    if total_docs <= 0:
        raise RuntimeError(
            "Normalized shard build produced zero documents. "
            "Check local dataset caching, source filters, or upstream availability."
        )

    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
