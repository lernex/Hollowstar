from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterator

import zstandard as zstd


def iter_jsonl_paths(
    *,
    jsonl_path: str | Path | None = None,
    jsonl_dir: str | Path | None = None,
    glob_pattern: str = "*.jsonl*",
) -> list[Path]:
    paths: list[Path] = []
    if jsonl_path is not None:
        paths.append(Path(jsonl_path))
    if jsonl_dir is not None:
        paths.extend(sorted(Path(jsonl_dir).glob(glob_pattern)))
    if not paths:
        raise ValueError("Expected at least one JSONL path or directory.")
    return paths


def open_jsonl_text_reader(path: str | Path):
    resolved = Path(path)
    handle = resolved.open("rb")
    if resolved.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(handle)
        text_stream = io.TextIOWrapper(stream, encoding="utf-8")
        return handle, stream, text_stream  # pragma: no cover
    text_stream = io.TextIOWrapper(handle, encoding="utf-8")
    return handle, None, text_stream  # pragma: no cover


def iter_jsonl_records(
    *,
    jsonl_path: str | Path | None = None,
    jsonl_dir: str | Path | None = None,
    glob_pattern: str = "*.jsonl*",
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    seen = 0
    for path in iter_jsonl_paths(jsonl_path=jsonl_path, jsonl_dir=jsonl_dir, glob_pattern=glob_pattern):
        raw_handle, binary_handle, text_handle = open_jsonl_text_reader(path)
        try:
            for line in text_handle:
                if max_rows is not None and seen >= max_rows:
                    return
                line = line.strip()
                if not line:
                    continue
                seen += 1
                yield json.loads(line)
        finally:
            text_handle.close()
            if binary_handle is not None:
                binary_handle.close()
            raw_handle.close()


def iter_jsonl_texts(
    *,
    jsonl_path: str | Path | None = None,
    jsonl_dir: str | Path | None = None,
    glob_pattern: str = "*.jsonl*",
    text_field: str = "text",
    max_rows: int | None = None,
) -> Iterator[str]:
    for row in iter_jsonl_records(
        jsonl_path=jsonl_path,
        jsonl_dir=jsonl_dir,
        glob_pattern=glob_pattern,
        max_rows=max_rows,
    ):
        text = row.get(text_field, "")
        if text:
            yield str(text)
