from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Regex, Tokenizer, normalizers

from .state import atomic_json, utc_now


CANONICAL_IDS_BINARY = "NGRAM_CANONICAL_IDS.uint16"
CANONICAL_IDS_MANIFEST = "NGRAM_CANONICAL_IDS.json"
CANONICAL_IDS_SCHEMA = "metis.ngram-canonical-ids/v1"
CANONICALIZATION_ALGORITHM = "deepseek-engram-normalization/v1"
CANONICALIZATION_REFERENCE = (
    "https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py"
)
_SENTINEL = "\uE000"
_NORMALIZATION_STEPS = (
    "NFKC",
    "NFD",
    "StripAccents",
    "Lowercase",
    "CollapseAsciiWhitespace",
    "PreserveSingleSpace",
    "Strip",
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalizer() -> normalizers.Sequence:
    # This is the exact compression normalization published in DeepSeek's
    # Engram reference. The sentinel prevents Strip() from erasing a token
    # whose normalized form is exactly one ASCII space.
    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), _SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(_SENTINEL, " "),
        ]
    )


def canonicalize_decoded_token(
    decoded_text: str,
    *,
    raw_token: str,
    tokenizer_normalizer: normalizers.Sequence | None = None,
) -> str:
    """Return the deterministic Engram compression key for one tokenizer ID."""

    if "\uFFFD" in decoded_text:
        return raw_token
    normalizer = tokenizer_normalizer or _normalizer()
    normalized = normalizer.normalize_str(decoded_text)
    return normalized if normalized else decoded_text


def canonical_ids_for_tokenizer(tokenizer: Tokenizer) -> np.ndarray:
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    ids = list(vocab.values())
    size = len(vocab)
    if size < 1 or len(set(ids)) != size or set(ids) != set(range(size)):
        raise RuntimeError("Tokenizer IDs must be unique and contiguous before canonicalization")
    if size > 65_536:
        raise RuntimeError(
            f"Canonical uint16 sidecar cannot represent a {size:,}-entry tokenizer"
        )

    tokenizer_normalizer = _normalizer()
    key_to_canonical_id: dict[str, int] = {}
    output = np.empty((size,), dtype="<u2")
    for token_id in range(size):
        raw_token = tokenizer.id_to_token(token_id)
        if raw_token is None:
            raise RuntimeError(f"Tokenizer has no raw token for ID {token_id}")
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        key = canonicalize_decoded_token(
            decoded,
            raw_token=raw_token,
            tokenizer_normalizer=tokenizer_normalizer,
        )
        canonical_id = key_to_canonical_id.get(key)
        if canonical_id is None:
            canonical_id = len(key_to_canonical_id)
            key_to_canonical_id[key] = canonical_id
        output[token_id] = canonical_id
    return output


def build_canonical_id_sidecar(
    tokenizer: Tokenizer,
    *,
    tokenizer_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build and atomically seal the tokenizer-ID compression sidecar."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_file = Path(tokenizer_path)
    canonical_ids = canonical_ids_for_tokenizer(tokenizer)
    binary_path = output / CANONICAL_IDS_BINARY
    temporary = output / f".{CANONICAL_IDS_BINARY}.tmp-{os.getpid()}"
    try:
        with temporary.open("wb") as handle:
            canonical_ids.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, binary_path)
    finally:
        temporary.unlink(missing_ok=True)

    unique_ids = np.unique(canonical_ids)
    canonical_vocabulary_size = int(unique_ids.size)
    if not np.array_equal(
        unique_ids.astype(np.int64, copy=False),
        np.arange(canonical_vocabulary_size, dtype=np.int64),
    ):
        raise RuntimeError("Canonical IDs are not contiguous from zero")
    descriptor: dict[str, Any] = {
        "schema": CANONICAL_IDS_SCHEMA,
        "created_at": utc_now(),
        "algorithm": CANONICALIZATION_ALGORITHM,
        "algorithm_reference": CANONICALIZATION_REFERENCE,
        "normalization_steps": list(_NORMALIZATION_STEPS),
        "replacement_character_fallback": "raw_tokenizer_vocab_token",
        "empty_normalization_fallback": "decoded_token",
        "tokenizer_sha256": _sha256(tokenizer_file),
        "vocabulary_size": int(canonical_ids.size),
        "entry_count": int(canonical_ids.size),
        "canonical_vocabulary_size": canonical_vocabulary_size,
        "minimum_canonical_id": int(canonical_ids.min(initial=0)),
        "maximum_canonical_id": int(canonical_ids.max(initial=0)),
        "canonical_ids_contiguous": True,
        "dtype": "uint16",
        "endianness": "little",
        "binary": CANONICAL_IDS_BINARY,
        "binary_size_bytes": int(binary_path.stat().st_size),
        "binary_sha256": _sha256(binary_path),
    }
    descriptor["manifest_sha256"] = _json_sha256(descriptor)
    atomic_json(output / CANONICAL_IDS_MANIFEST, descriptor)
    return descriptor


def validate_canonical_id_sidecar(
    *,
    manifest_path: str | Path,
    binary_path: str | Path,
    tokenizer_path: str | Path,
    expected_vocabulary_size: int,
    expected_manifest_sha256: str | None = None,
    expected_binary_sha256: str | None = None,
    recompute_from_tokenizer: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    """Validate canonical-ID bytes, lineage, and optionally exact semantics."""

    manifest_file = Path(manifest_path)
    binary_file = Path(binary_path)
    tokenizer_file = Path(tokenizer_path)
    for label, path in (
        ("canonical-ID manifest", manifest_file),
        ("canonical-ID binary", binary_file),
        ("tokenizer", tokenizer_file),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{label} is missing or a symlink: {path}")

    descriptor = json.loads(manifest_file.read_text(encoding="utf-8"))
    unsigned = {
        key: value for key, value in descriptor.items() if key != "manifest_sha256"
    }
    manifest_sha256 = _json_sha256(unsigned)
    tokenizer_sha256 = _sha256(tokenizer_file)
    binary_sha256 = _sha256(binary_file)
    expected_size = int(expected_vocabulary_size)
    if (
        descriptor.get("schema") != CANONICAL_IDS_SCHEMA
        or descriptor.get("algorithm") != CANONICALIZATION_ALGORITHM
        or descriptor.get("normalization_steps") != list(_NORMALIZATION_STEPS)
        or descriptor.get("replacement_character_fallback")
        != "raw_tokenizer_vocab_token"
        or descriptor.get("empty_normalization_fallback") != "decoded_token"
        or descriptor.get("manifest_sha256") != manifest_sha256
        or (
            expected_manifest_sha256 is not None
            and manifest_sha256 != expected_manifest_sha256
        )
        or descriptor.get("tokenizer_sha256") != tokenizer_sha256
        or int(descriptor.get("vocabulary_size", -1)) != expected_size
        or int(descriptor.get("entry_count", -1)) != expected_size
        or descriptor.get("dtype") != "uint16"
        or descriptor.get("endianness") != "little"
        or descriptor.get("binary") != binary_file.name
        or int(descriptor.get("binary_size_bytes", -1)) != expected_size * 2
        or binary_file.stat().st_size != expected_size * 2
        or descriptor.get("binary_sha256") != binary_sha256
        or (
            expected_binary_sha256 is not None
            and binary_sha256 != expected_binary_sha256
        )
        or descriptor.get("canonical_ids_contiguous") is not True
    ):
        raise RuntimeError("Canonical-ID sidecar descriptor or lineage is invalid")

    canonical_ids = np.fromfile(binary_file, dtype="<u2")
    if canonical_ids.shape != (expected_size,):
        raise RuntimeError(
            "Canonical-ID sidecar shape is invalid: "
            f"expected {(expected_size,)}, observed {canonical_ids.shape}"
        )
    unique_ids = np.unique(canonical_ids)
    canonical_vocabulary_size = int(descriptor.get("canonical_vocabulary_size", -1))
    if (
        canonical_vocabulary_size < 1
        or canonical_vocabulary_size > expected_size
        or unique_ids.shape != (canonical_vocabulary_size,)
        or not np.array_equal(
            unique_ids.astype(np.int64, copy=False),
            np.arange(canonical_vocabulary_size, dtype=np.int64),
        )
        or int(descriptor.get("minimum_canonical_id", -1)) != int(canonical_ids.min())
        or int(descriptor.get("maximum_canonical_id", -1)) != int(canonical_ids.max())
    ):
        raise RuntimeError("Canonical-ID sidecar values are not a contiguous surjection")

    if recompute_from_tokenizer:
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        expected = canonical_ids_for_tokenizer(tokenizer)
        if not np.array_equal(canonical_ids, expected):
            raise RuntimeError(
                "Canonical-ID sidecar does not implement the declared tokenizer normalization"
            )
    return descriptor, canonical_ids
