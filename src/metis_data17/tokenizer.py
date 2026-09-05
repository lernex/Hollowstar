from __future__ import annotations

import array
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from metis_data.tokenizer import tokenizer_splits_digits

from .common import atomic_json, canonical_json, digest_json, read_receipt, sha256_file, utc_now


PRODUCTION_VOCABULARY_SIZE = 131_072
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
TOKENIZER_RELEASE = "TOKENIZER_RELEASE.json"
DIGIT_POLICY = {"type": "Digits", "individual_digits": True, "before_byte_level": True}

PREPARED_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("content_hash", pa.string()),
        ("dedup_hash", pa.string()),
        ("source_id", pa.string()),
        ("object_id", pa.string()),
        ("text", pa.string()),
        ("metadata_json", pa.string()),
        ("priority", pa.int32()),
        ("quality_score", pa.float64()),
        ("language", pa.string()),
        ("category", pa.string()),
        ("character_count", pa.int64()),
    ]
)
CACHE_SCHEMA = pa.schema(
    [
        ("content_hash", pa.string()),
        ("cache_key", pa.string()),
        ("token_offset", pa.int64()),
        ("token_count", pa.int64()),
        ("ids_sha256", pa.string()),
    ]
)
TOKENIZED_SCHEMA = pa.schema(
    [
        ("source_shard", pa.string()),
        ("source_row", pa.int64()),
        *[field for field in PREPARED_SCHEMA if field.name != "text"],
        ("cache_key", pa.string()),
        ("ids_path", pa.string()),
        ("token_offset", pa.int64()),
        ("token_count", pa.int64()),
    ]
)
SAMPLE_SCHEMA = pa.schema(
    [
        ("source_number", pa.int32()),
        ("source_shard", pa.string()),
        ("source_row", pa.int64()),
        ("doc_id", pa.string()),
        ("content_hash", pa.string()),
        ("stratum", pa.string()),
        ("rank", pa.string()),
        ("utf8_bytes", pa.int64()),
    ]
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LOCAL_FILESYSTEMS = {"apfs", "hfs", "hfsplus", "ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs", "ufs", "tmpfs", "ramfs"}


@dataclass(frozen=True)
class TokenCacheLimits17:
    max_documents: int = 250_000
    max_shards: int = 2_048
    max_token_bytes: int = 32 * 1024**3
    max_scratch_bytes: int = 512 * 1024**2
    max_input_paths: int = 4_096

    def validate(self) -> None:
        for name, value in asdict(self).items():
            _positive_integer(value, name)
        if self.max_scratch_bytes < 256 * 1024:
            raise ValueError("max_scratch_bytes must allow at least 256 KiB")


def _filesystem_type17(path: Path) -> str:
    if sys.platform == "linux":
        def unescape(value: str) -> str:
            return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)

        mounts = []
        with Path("/proc/self/mountinfo").open(encoding="utf-8") as stream:
            for line in stream:
                before, after = line.rstrip().split(" - ", 1)
                mount = Path(unescape(before.split()[4]))
                if path.is_relative_to(mount):
                    mounts.append((len(mount.parts), after.split()[0]))
        return max(mounts)[1] if mounts else "unknown"
    if sys.platform == "darwin":
        # Allow posix_spawn rather than forking an initialized tokenizers Rayon pool.
        result = subprocess.run(["/sbin/mount"], capture_output=True, text=True, check=True, close_fds=False)
        mounts = []
        for line in result.stdout.splitlines():
            match = re.match(r".* on (.+) \(([^,)]+)(?:,.*)?\)$", line)
            if match and path.is_relative_to(Path(match[1])):
                mounts.append((len(Path(match[1]).parts), match[2]))
        return max(mounts)[1] if mounts else "unknown"
    return "unknown"


def _local_scratch(scratch_dir: Path, output: Path, budget: int) -> Path:
    _positive_integer(budget, "max_scratch_bytes")
    if budget < 256 * 1024:
        raise ValueError("max_scratch_bytes must allow at least 256 KiB")
    scratch = Path(scratch_dir).expanduser().resolve()
    if scratch.is_relative_to(output) or output.is_relative_to(scratch):
        raise ValueError("Node-local scratch and durable output must be disjoint directories")
    ancestor = scratch
    while not ancestor.exists():
        ancestor = ancestor.parent
    if _filesystem_type17(ancestor) not in _LOCAL_FILESYSTEMS:
        raise ValueError("scratch_dir must be on a verified node-local filesystem, never Lustre/NFS/unknown")
    scratch.mkdir(parents=True, exist_ok=True)
    if _filesystem_type17(scratch) not in _LOCAL_FILESYSTEMS:
        raise ValueError("scratch_dir resolved onto a non-local filesystem")
    if shutil.disk_usage(scratch).free < budget:
        raise ValueError("Insufficient node-local free space for the explicit scratch budget")
    return scratch


def _local_database(path: Path, budget: int) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        path.parent.is_symlink()
        or any(Path(str(path) + suffix).is_symlink() for suffix in ("", "-journal", "-wal", "-shm"))
        or _filesystem_type17(path.parent.resolve()) not in _LOCAL_FILESYSTEMS
    ):
        raise ValueError("SQLite files must remain directly on verified node-local scratch")
    database = sqlite3.connect(path)
    try:
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA journal_mode=DELETE")
        database.execute("PRAGMA synchronous=FULL")
        database.execute("PRAGMA cache_size=-8192")
        database.execute("PRAGMA temp_store=MEMORY")
        page_size = database.execute("PRAGMA page_size").fetchone()[0]
        # Reserve space for a worst-case rollback journal and its page headers.
        pages = (budget - 64 * 1024) // (2 * (page_size + 16))
        if database.execute("PRAGMA page_count").fetchone()[0] > pages:
            raise ValueError("Rebuildable SQLite cache exceeds the node-local scratch limit")
        database.execute(f"PRAGMA max_page_count={pages}")
        return database
    except Exception:
        database.close()
        raise


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"Invalid {name}: expected lowercase SHA-256")
    return value


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _sealed(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = digest_json(result)
    atomic_json(path, result)
    _sync_directory(path.parent)
    return result


def _read_sealed(path: Path) -> dict[str, Any]:
    try:
        result = read_receipt(path)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid or missing receipt: {path}: {error}") from error
    result["receipt_sha256"] = digest_json(result)
    return result


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


@contextlib.contextmanager
def _locked(path: Path, *, blocking: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise ValueError(f"Worker/partition lock is already held: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _stage(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / name
    if path.is_symlink():
        raise ValueError(f"Refusing symlink staging directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Invalid staging directory: {path}")
        shutil.rmtree(path)
    path.mkdir()
    return path


def _publish(stage: Path, destination: Path) -> None:
    _sync_directory(stage)
    os.replace(stage, destination)
    _sync_directory(destination.parent)
    if stage.parent != destination.parent:
        _sync_directory(stage.parent)


def _new_tokenizer() -> Tokenizer:
    # A complete ByteLevel alphabet covers every UTF-8 byte without an UNK path.
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def _validate_tokenizer(tokenizer: Tokenizer, *, production: bool) -> dict[str, int]:
    payload = json.loads(tokenizer.to_str())
    if not tokenizer_splits_digits(tokenizer):
        raise ValueError("Metis 1.7 requires global individual digit splitting")
    expected = json.loads(_new_tokenizer().to_str())
    for component in ("normalizer", "pre_tokenizer", "decoder", "post_processor", "padding", "truncation"):
        if payload.get(component) != expected.get(component):
            raise ValueError(f"Unsupported tokenizer {component}; exact lossless digit policy is required")
    model = payload.get("model", {})
    if (
        model.get("type") != "BPE"
        or model.get("unk_token") is not None
        or model.get("dropout") is not None
        or model.get("continuing_subword_prefix")
        or model.get("end_of_word_suffix")
    ):
        raise ValueError("Tokenizer must use deterministic, lossless byte-level BPE")
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    ids = set(vocabulary.values())
    if ids != set(range(len(vocabulary))) or len(vocabulary) > 2**32:
        raise ValueError("Tokenizer IDs must be dense and fit uint32")
    if production and len(vocabulary) != PRODUCTION_VOCABULARY_SIZE:
        raise ValueError(f"Production tokenizer requires exactly 131072 entries, got {len(vocabulary)}")
    if not set(pre_tokenizers.ByteLevel.alphabet()).issubset(model.get("vocab", {})):
        raise ValueError("Tokenizer is missing the complete ByteLevel alphabet")
    for token in payload.get("added_tokens", []):
        if (
            token.get("special") is not True
            or token.get("lstrip") is not False
            or token.get("rstrip") is not False
            or token.get("single_word") is not False
            or token.get("normalized") is not False
        ):
            raise ValueError("Added tokens must be literal, non-normalizing special tokens")
        if tokenizer.decode([token["id"]], skip_special_tokens=False) != token["content"]:
            raise ValueError("Special token does not round-trip through the ByteLevel decoder")
    return vocabulary


def train_tokenizer17(
    texts: Iterable[str],
    output_dir: Path,
    *,
    vocabulary_size: int = PRODUCTION_VOCABULARY_SIZE,
    special_tokens: list[str] | None = None,
    minimum_frequency: int = 2,
    production: bool = True,
) -> dict[str, Any]:
    """Train once and atomically freeze an artifact; use load_tokenizer17 to reuse it."""
    if type(production) is not bool:
        raise ValueError("production must be a boolean")
    _positive_integer(vocabulary_size, "vocabulary_size")
    _positive_integer(minimum_frequency, "minimum_frequency")
    if special_tokens is not None and not isinstance(special_tokens, list):
        raise ValueError("special_tokens must be a list of literal strings")
    specials = list(DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens)
    if not specials or any(not isinstance(token, str) or not token for token in specials):
        raise ValueError("special_tokens must contain nonempty strings, with EOS first")
    if len(set(specials)) != len(specials):
        raise ValueError("special_tokens must be unique")
    if any(any(character.isnumeric() for character in token) for token in specials):
        raise ValueError("Special tokens cannot bypass global splitting by containing digits")
    if vocabulary_size < 256 + len(specials) or vocabulary_size > 2**32:
        raise ValueError("vocabulary_size must accommodate the byte alphabet and special tokens in uint32")
    if production and vocabulary_size != PRODUCTION_VOCABULARY_SIZE:
        raise ValueError("Production vocabulary_size must be exactly 131072")
    output = Path(output_dir).resolve()
    with _locked(output.parent / f".{output.name}.training.lock"):
        if output.exists() and any(output.iterdir()):
            if (output / TOKENIZER_RELEASE).exists():
                load_tokenizer17(output, production=production)
            raise FileExistsError(f"Tokenizer artifacts are immutable; choose a new output directory: {output}")
        stage = _stage(output.parent, f".{output.name}.training-incomplete")
        try:
            tokenizer = _new_tokenizer()
            training_hash = hashlib.sha256()
            counts = {"documents": 0, "utf8_bytes": 0}

            def checked_texts() -> Iterator[str]:
                for text in texts:
                    if not isinstance(text, str):
                        raise ValueError("Tokenizer training requires prepared string texts")
                    encoded = text.encode("utf-8")
                    training_hash.update(len(encoded).to_bytes(8, "little"))
                    training_hash.update(encoded)
                    counts["documents"] += 1
                    counts["utf8_bytes"] += len(encoded)
                    yield text

            trainer = trainers.BpeTrainer(
                vocab_size=vocabulary_size,
                min_frequency=minimum_frequency,
                special_tokens=specials,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                show_progress=False,
            )
            tokenizer.train_from_iterator(checked_texts(), trainer=trainer, length=None)
            vocabulary = _validate_tokenizer(tokenizer, production=production)
            if counts["utf8_bytes"] == 0:
                raise ValueError("Tokenizer training corpus is empty")
            if len(vocabulary) > vocabulary_size:
                raise ValueError("Tokenizer exceeded the requested vocabulary size")
            tokenizer_path = stage / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            _sync_file(tokenizer_path)
            release = _sealed(
                stage / TOKENIZER_RELEASE,
                {
                    "schema": "metis17.tokenizer/v1",
                    "created_at": utc_now(),
                    "production": production,
                    "algorithm": "byte_level_bpe",
                    "split_digits": True,
                    "digit_policy": dict(DIGIT_POLICY),
                    "digit_policy_sha256": digest_json(DIGIT_POLICY),
                    "dtype": "<u4",
                    "byte_order": "little",
                    "vocabulary_size": len(vocabulary),
                    "requested_vocabulary_size": vocabulary_size,
                    "maximum_token_id": max(vocabulary.values()),
                    "special_tokens": {token: tokenizer.token_to_id(token) for token in specials},
                    "eos_token": specials[0],
                    "eos_token_id": tokenizer.token_to_id(specials[0]),
                    "minimum_frequency": minimum_frequency,
                    "tokenizer_sha256": sha256_file(tokenizer_path),
                    "tokenizers_version": tokenizers.__version__,
                    "training": {**counts, "ordered_text_sha256": training_hash.hexdigest()},
                },
            )
            load_tokenizer17(stage, production=production)
            _publish(stage, output)
            return release
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def _load_artifact(directory: Path, *, production: bool) -> tuple[Tokenizer, dict[str, Any]]:
    if type(production) is not bool:
        raise ValueError("production must be a boolean")
    directory = Path(directory)
    release = _read_sealed(directory / TOKENIZER_RELEASE)
    if release.get("schema") != "metis17.tokenizer/v1":
        raise ValueError("Unsupported tokenizer release schema")
    if type(release.get("production")) is not bool or (production and not release["production"]):
        raise ValueError("A test tokenizer cannot be loaded for production")
    if (
        release.get("split_digits") is not True
        or release.get("digit_policy") != DIGIT_POLICY
        or release.get("digit_policy_sha256") != digest_json(DIGIT_POLICY)
        or release.get("dtype") != "<u4"
        or release.get("byte_order") != "little"
        or release.get("algorithm") != "byte_level_bpe"
    ):
        raise ValueError("Tokenizer release has a mismatched digit or uint32 policy")
    path = directory / "tokenizer.json"
    expected_hash = _digest(release.get("tokenizer_sha256"), "tokenizer_sha256")
    if not path.is_file():
        raise ValueError("Missing tokenizer artifact")
    serialized = path.read_bytes()
    if hashlib.sha256(serialized).hexdigest() != expected_hash:
        raise ValueError("Tokenizer artifact hash mismatch")
    try:
        tokenizer = Tokenizer.from_str(serialized.decode("utf-8"))
        vocabulary = _validate_tokenizer(tokenizer, production=release["production"])
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"Invalid tokenizer artifact: {error}") from error
    requested = release.get("requested_vocabulary_size")
    if (
        type(requested) is not int
        or requested < len(vocabulary)
        or requested > 2**32
        or (release["production"] and requested != PRODUCTION_VOCABULARY_SIZE)
        or type(release.get("vocabulary_size")) is not int
        or type(release.get("maximum_token_id")) is not int
        or release.get("vocabulary_size") != len(vocabulary)
        or release.get("maximum_token_id") != max(vocabulary.values())
    ):
        raise ValueError("Tokenizer release vocabulary does not match the artifact")
    specials = release.get("special_tokens")
    actual_specials = {
        entry["content"]: entry["id"] for entry in json.loads(tokenizer.to_str())["added_tokens"]
    }
    if not isinstance(specials, dict) or not specials or specials != actual_specials:
        raise ValueError("Tokenizer release special tokens do not match the artifact")
    if any(any(character.isnumeric() for character in token) for token in specials):
        raise ValueError("Special tokens cannot override global digit splitting")
    if any(vocabulary.get(token) != token_id or type(token_id) is not int for token, token_id in specials.items()):
        raise ValueError("Tokenizer release special token IDs are invalid")
    if (
        release.get("eos_token") not in specials
        or type(release.get("eos_token_id")) is not int
        or specials[release["eos_token"]] != release.get("eos_token_id")
    ):
        raise ValueError("Tokenizer release EOS does not match the artifact")
    return tokenizer, release


def load_tokenizer17(directory: Path, *, production: bool = True) -> Tokenizer:
    """Verify the self-hashed release, serialized digit policy, and actual vocabulary."""
    return _load_artifact(directory, production=production)[0]


def tokenization_policy17() -> dict[str, Any]:
    return {
        "schema": "metis17.raw-token-ids/v1",
        "tokenizers_version": tokenizers.__version__,
        "digit_policy": dict(DIGIT_POLICY),
        "add_special_tokens": False,
        "normalization": "none",
        "padding": False,
        "truncation": False,
        "text_encoding": "utf-8",
        "dtype": "<u4",
        "offset_unit": "tokens",
    }


def token_cache_key17(content_hash: str, tokenizer_sha256: str, policy: Mapping[str, Any] | None = None) -> str:
    return digest_json(
        {
            "content_hash": _digest(content_hash, "content_hash"),
            "tokenizer_sha256": _digest(tokenizer_sha256, "tokenizer_sha256"),
            "tokenization_policy_sha256": digest_json(tokenization_policy17() if policy is None else policy),
        }
    )


def _prepared_file(path: Path) -> pq.ParquetFile:
    if any(part.lower() in {"raw", "quarantine", "quarantined"} for part in path.parts):
        raise ValueError(f"Only eligible prepared Parquets may be tokenized or sampled: {path}")
    parquet = pq.ParquetFile(path)
    for field in PREPARED_SCHEMA:
        if field.name not in parquet.schema_arrow.names or parquet.schema_arrow.field(field.name).type != field.type:
            raise ValueError(f"Prepared Parquet is missing or has an invalid {field.name} field: {path}")
    if len(parquet.schema_arrow.names) != len(set(parquet.schema_arrow.names)):
        raise ValueError(f"Duplicate fields in prepared Parquet: {path}")
    return parquet


def _input_snapshot(input_paths: Sequence[Path]) -> list[dict[str, Any]]:
    paths = sorted(Path(path).resolve() for path in input_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Supply nonempty, distinct eligible prepared Parquet paths")
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
            "rows": _prepared_file(path).metadata.num_rows,
        }
        for path in paths
    ]


def _unchanged(source: Mapping[str, Any]) -> None:
    path = Path(source["path"])
    if path.stat().st_size != source["byte_count"] or sha256_file(path) != source["sha256"]:
        raise ValueError(f"Prepared source changed during processing: {path}")


def _checked_row(row: Mapping[str, Any]) -> int:
    if any(row.get(field.name) is None for field in PREPARED_SCHEMA):
        raise ValueError("Prepared records cannot contain null required fields")
    text = row["text"]
    if not isinstance(text, str):
        raise ValueError("Prepared text must be a string")
    encoded = text.encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != _digest(row["content_hash"], "content_hash"):
        raise ValueError("Prepared content_hash does not match exact UTF-8 text")
    if row["character_count"] != len(text):
        raise ValueError("Prepared character_count does not match exact text")
    if not math.isfinite(row["quality_score"]):
        raise ValueError("Prepared quality_score must be finite (use -1 for absent scores)")
    return len(encoded)


def _checked_metadata(row: Mapping[str, Any]) -> None:
    if any(row.get(field.name) is None for field in PREPARED_SCHEMA if field.name != "text"):
        raise ValueError("Prepared metadata cannot contain null required fields")
    _digest(row["content_hash"], "content_hash")
    if type(row["character_count"]) is not int or row["character_count"] < 0:
        raise ValueError("Prepared character_count must be a nonnegative integer")
    if not math.isfinite(row["quality_score"]):
        raise ValueError("Prepared quality_score must be finite")


def _candidate_sizes(
    inputs: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]], batch_size: int,
) -> dict[str, int]:
    references = iter(sorted(candidates, key=lambda row: (row["source_number"], row["source_row"])))
    pending = next(references, None)
    sizes = {}
    while pending is not None:
        source_number = pending["source_number"]
        parquet = _prepared_file(Path(inputs[source_number]["path"]))
        group_start = 0
        for group_number in range(parquet.metadata.num_row_groups):
            group_end = group_start + parquet.metadata.row_group(group_number).num_rows
            if pending is not None and pending["source_number"] == source_number and pending["source_row"] < group_end:
                source_row = group_start
                for batch in parquet.iter_batches(
                    batch_size=batch_size, row_groups=[group_number],
                    columns=["text", "doc_id", "content_hash", "character_count"],
                ):
                    for row in batch.to_pylist():
                        if pending is not None and pending["source_number"] == source_number and pending["source_row"] == source_row:
                            data = row["text"].encode("utf-8")
                            if (
                                row["doc_id"] != pending["doc_id"]
                                or row["content_hash"] != pending["content_hash"]
                                or len(row["text"]) != pending["character_count"]
                                or hashlib.sha256(data).hexdigest() != pending["content_hash"]
                            ):
                                raise ValueError("Sample candidate does not match its prepared text")
                            sizes[pending["content_hash"]] = len(data)
                            pending = next(references, None)
                        source_row += 1
                    if pending is None or pending["source_number"] != source_number or pending["source_row"] >= group_end:
                        break
            group_start = group_end
        if pending is not None and pending["source_number"] == source_number:
            raise ValueError("Missing sampled source row")
    return sizes


def _parquet_rows(path: Path, *, batch_size: int = 256) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def _ids_bytes(ids: Sequence[int]) -> bytes:
    result = array.array("I", ids)
    if result.itemsize != 4:
        raise RuntimeError("This platform does not provide 32-bit unsigned integer arrays")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


class _TokenCache:
    def __init__(
        self, output: Path, tokenizer_sha256: str, policy: Mapping[str, Any],
        *, scratch: Path, partition_id: str, limits: TokenCacheLimits17,
    ) -> None:
        self.output = output
        self.tokenizer_sha256 = tokenizer_sha256
        self.policy = dict(policy)
        self.policy_sha256 = digest_json(policy)
        self.partition_id = partition_id
        self.limits = limits
        self.root = output / "cache" / tokenizer_sha256 / self.policy_sha256 / "partitions" / partition_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards = self.root / "shards"
        self.shards.mkdir(exist_ok=True)
        self.commits = self.root / "commits"
        self.commits.mkdir(exist_ok=True)
        self.verified: dict[str, tuple[Any, ...]] = {}
        self.seen_this_call: set[str] = set()
        self._checked_miss_count = False
        local_identity = {"output": str(output), "identity": self.identity}
        local = scratch / "token-cache"
        if local.is_symlink():
            raise ValueError("Node-local cache directory cannot be a symlink")
        marker = local / "IDENTITY.json"
        if not marker.exists() or _read_sealed(marker).get("cache") != local_identity:
            _stage(scratch, "token-cache")
            _sealed(marker, {"cache": local_identity})
        self.database_path = local / "index.sqlite3"
        self.database = _local_database(self.database_path, limits.max_scratch_bytes)
        try:
            self.database.execute("PRAGMA foreign_keys=ON")
            self.database.executescript(
                """
                CREATE TABLE IF NOT EXISTS header (identity TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoint (sequence INTEGER NOT NULL, commit_sha256 TEXT);
                CREATE TABLE IF NOT EXISTS shards (
                    shard_id TEXT PRIMARY KEY, receipt_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entries (
                    content_hash TEXT PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE,
                    shard_id TEXT NOT NULL REFERENCES shards(shard_id),
                    token_offset INTEGER NOT NULL, token_count INTEGER NOT NULL,
                    ids_sha256 TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS entries_by_shard ON entries(shard_id);
                """
            )
            identity = canonical_json(self.identity)
            headers = self.database.execute("SELECT identity FROM header").fetchall()
            if not headers:
                self.database.execute("INSERT INTO header VALUES (?)", (identity,))
                self.database.execute("INSERT INTO checkpoint VALUES (0, NULL)")
            elif len(headers) != 1 or headers[0]["identity"] != identity:
                raise ValueError("Mismatched token-cache database identity")
            self.database.commit()
            self._recover()
        except Exception:
            self.database.close()
            raise

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema": "metis17.token-cache/v2",
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenization_policy": self.policy,
            "tokenization_policy_sha256": self.policy_sha256,
            "partition_id": self.partition_id,
            "limits": {
                "max_documents": self.limits.max_documents,
                "max_shards": self.limits.max_shards,
                "max_token_bytes": self.limits.max_token_bytes,
            },
        }

    def close(self) -> None:
        self.database.close()

    def _receipt(self, shard: Path) -> dict[str, Any]:
        receipt = _read_sealed(shard / "receipt.json")
        if (
            receipt.get("schema") != "metis17.token-cache-shard/v1"
            or receipt.get("identity") != self.identity
            or receipt.get("shard_id") != shard.name
        ):
            raise ValueError(f"Mismatched cached ID shard identity: {shard}")
        return receipt

    def _head(self) -> dict[str, Any]:
        head = _read_sealed(self.root / "HEAD.json")
        if head.get("identity") != self.identity:
            raise ValueError("Mismatched cache partition identity or admission limits")
        for field, limit in (
            ("sequence", self.limits.max_shards),
            ("documents", self.limits.max_documents),
            ("token_bytes", self.limits.max_token_bytes),
        ):
            if type(head.get(field)) is not int or not 0 <= head[field] <= limit:
                raise ValueError("Cache partition exceeds its explicit bounded limits")
        return head

    def _finish_pending(self) -> None:
        pending_path = self.root / "PENDING.json"
        if not pending_path.exists():
            return
        pending = _read_sealed(pending_path)
        if pending.get("identity") != self.identity:
            raise ValueError("Mismatched interrupted cache publication")
        for field, limit in (
            ("sequence", self.limits.max_shards), ("documents", self.limits.max_documents),
            ("token_bytes", self.limits.max_token_bytes),
        ):
            if type(pending.get(field)) is not int or not 0 <= pending[field] <= limit:
                raise ValueError("Interrupted publication exceeds partition limits")
        shard_id = _digest(pending["shard_id"], "pending shard")
        destination = self.shards / shard_id
        if not destination.exists():
            _publish(self.root / ".incomplete-shard", destination)
        receipt = self._receipt(destination)
        if receipt["receipt_sha256"] != pending["shard_receipt_sha256"]:
            raise ValueError("Interrupted cache shard receipt mismatch")
        head = self._head()
        sequence = pending["sequence"]
        commit_path = self.commits / f"{sequence:012d}.json"
        payload = {
            "identity": self.identity, "sequence": sequence,
            "previous_commit_sha256": pending["previous_commit_sha256"],
            "shard_id": shard_id, "shard_receipt_sha256": receipt["receipt_sha256"],
            "documents": pending["documents"], "token_bytes": pending["token_bytes"],
        }
        if commit_path.exists():
            commit = _read_sealed(commit_path)
            if {key: value for key, value in commit.items() if key != "receipt_sha256"} != payload:
                raise ValueError("Conflicting immutable cache commit")
        else:
            commit = _sealed(commit_path, payload)
        if head["sequence"] == sequence - 1:
            if (
                head["commit_sha256"] != pending["previous_commit_sha256"]
                or head["documents"] + receipt["records"] != pending["documents"]
                or head["token_bytes"] + receipt["byte_count"] != pending["token_bytes"]
            ):
                raise ValueError("Interrupted publication has inconsistent partition totals")
            _sealed(self.root / "HEAD.json", {
                "identity": self.identity, "sequence": sequence,
                "commit_sha256": commit["receipt_sha256"],
                "documents": pending["documents"], "token_bytes": pending["token_bytes"],
            })
        elif head["sequence"] != sequence or head["commit_sha256"] != commit["receipt_sha256"]:
            raise ValueError("Interrupted publication conflicts with partition head")
        pending_path.unlink()
        _sync_directory(self.root)

    def _recover(self) -> None:
        head_path = self.root / "HEAD.json"
        if not head_path.exists():
            if (self.commits / "000000000001.json").exists():
                raise ValueError("Missing committed cache partition head")
            _sealed(head_path, {
                "identity": self.identity, "sequence": 0, "commit_sha256": None,
                "documents": 0, "token_bytes": 0,
            })
        self._finish_pending()
        incomplete = self.root / ".incomplete-shard"
        if incomplete.exists():
            if incomplete.is_symlink() or not incomplete.is_dir():
                raise ValueError("Invalid interrupted cache staging directory")
            shutil.rmtree(incomplete)
        self.head = self._head()
        checkpoints = self.database.execute("SELECT * FROM checkpoint").fetchall()
        if len(checkpoints) != 1:
            raise ValueError("Corrupt local cache checkpoint")
        sequence, previous = checkpoints[0]["sequence"], checkpoints[0]["commit_sha256"]
        if type(sequence) is not int or not 0 <= sequence <= self.head["sequence"]:
            raise ValueError("Local cache checkpoint is ahead of durable publication")
        # Only replay new commits in this bounded partition. A warm index needs one HEAD read.
        for number in range(sequence + 1, self.head["sequence"] + 1):
            commit = _read_sealed(self.commits / f"{number:012d}.json")
            if (
                commit.get("identity") != self.identity
                or commit.get("sequence") != number
                or commit.get("previous_commit_sha256") != previous
            ):
                raise ValueError("Broken immutable cache commit chain")
            shard_id = _digest(commit["shard_id"], "committed shard")
            receipt = self._receipt(self.shards / shard_id)
            if receipt["receipt_sha256"] != commit["shard_receipt_sha256"]:
                raise ValueError("Committed cache shard receipt mismatch")
            with self.database:
                self._register(shard_id, receipt)
                self.database.execute("UPDATE checkpoint SET sequence=?, commit_sha256=?", (number, commit["receipt_sha256"]))
            previous = commit["receipt_sha256"]
        if previous != self.head["commit_sha256"]:
            raise ValueError("Local cache checkpoint disagrees with durable partition head")

    def _validated_entries(
        self, shard_id: str, receipt: Mapping[str, Any], *, verify_ids: bool = True,
    ) -> Iterator[dict[str, Any]]:
        shard = self.shards / shard_id
        binary = shard / "ids.bin"
        offsets = shard / "offsets.parquet"
        if not binary.is_file() or binary.is_symlink():
            raise ValueError(f"Corrupt cached artifact: {binary}")
        if not offsets.is_file() or offsets.is_symlink() or sha256_file(offsets) != receipt.get("offsets_sha256"):
            raise ValueError(f"Corrupt cached artifact: {offsets}")
        if pq.ParquetFile(offsets).schema_arrow != CACHE_SCHEMA:
            raise ValueError("Corrupt cached offset schema")
        offset = 0
        count = 0
        hashes = hashlib.sha256()
        binary_hash = hashlib.sha256()
        with (binary.open("rb") if verify_ids else contextlib.nullcontext()) as stream:
            for row in _parquet_rows(offsets):
                if (
                    row["cache_key"] != token_cache_key17(row["content_hash"], self.tokenizer_sha256, self.policy)
                    or type(row["token_count"]) is not int
                    or row["token_count"] < 0
                    or row["token_offset"] != offset
                ):
                    raise ValueError("Corrupt cached offsets, counts, or cache keys")
                record_hash = hashlib.sha256()
                remaining = row["token_count"] * 4
                while remaining and verify_ids:
                    block = stream.read(min(remaining, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError("Corrupt cached artifact: truncated uint32 IDs")
                    binary_hash.update(block)
                    record_hash.update(block)
                    remaining -= len(block)
                _digest(row["ids_sha256"], "record IDs digest")
                if verify_ids and record_hash.hexdigest() != row["ids_sha256"]:
                    raise ValueError("Corrupt cached artifact: record ID checksum mismatch")
                hashes.update(bytes.fromhex(row["content_hash"]))
                offset += row["token_count"]
                count += 1
                yield row
            if verify_ids and stream.read(1):
                raise ValueError("Corrupt cached artifact: unreferenced uint32 IDs")
        if (
            offset * 4 != binary.stat().st_size
            or (verify_ids and binary_hash.hexdigest() != receipt.get("ids_sha256"))
            or offset != receipt.get("token_count")
            or count != receipt.get("records")
            or hashes.hexdigest() != shard_id
            or receipt.get("byte_count") != offset * 4
        ):
            raise ValueError("Cached shard has gaps, missing records, or invalid uint32 length")

    def _register(self, shard_id: str, receipt: Mapping[str, Any]) -> None:
        try:
            self.database.execute(
                "INSERT INTO shards VALUES (?, ?)", (shard_id, receipt["receipt_sha256"]),
            )
            for row in self._validated_entries(shard_id, receipt, verify_ids=False):
                self.database.execute(
                    "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?)",
                    (row["content_hash"], row["cache_key"], shard_id, row["token_offset"], row["token_count"], row["ids_sha256"]),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("Duplicate or conflicting token-cache entries") from error

    def _verify(self, shard_id: str) -> None:
        if shard_id in self.seen_this_call:
            return
        shard = self.shards / shard_id
        try:
            stamp = tuple(
                (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                for stat in ((shard / name).stat() for name in ("ids.bin", "offsets.parquet", "receipt.json"))
            )
        except OSError as error:
            raise ValueError(f"Missing committed cache artifact: {shard}: {error}") from error
        if self.verified.get(shard_id) == stamp:
            self.seen_this_call.add(shard_id)
            return
        known = self.database.execute("SELECT * FROM shards WHERE shard_id=?", (shard_id,)).fetchone()
        if known is None:
            raise ValueError("Cache index references an unknown shard")
        receipt = self._receipt(self.shards / shard_id)
        if known["receipt_sha256"] != receipt["receipt_sha256"]:
            raise ValueError("Cached shard receipt changed after publication")
        count = 0
        for row in self._validated_entries(shard_id, receipt):
            indexed = self.database.execute("SELECT * FROM entries WHERE content_hash=?", (row["content_hash"],)).fetchone()
            if indexed is None or indexed["shard_id"] != shard_id or any(indexed[key] != value for key, value in row.items()):
                raise ValueError("Token-cache index disagrees with committed offsets")
            count += 1
        if self.database.execute("SELECT COUNT(*) FROM entries WHERE shard_id=?", (shard_id,)).fetchone()[0] != count:
            raise ValueError("Token-cache index contains extra records")
        self.verified[shard_id] = stamp
        self.seen_this_call.add(shard_id)

    def lookup(self, content_hash: str) -> dict[str, Any] | None:
        row = self.database.execute("SELECT * FROM entries WHERE content_hash=?", (content_hash,)).fetchone()
        if row is None:
            if not self._checked_miss_count:
                if (
                    self.database.execute("SELECT COUNT(*) FROM entries").fetchone()[0] != self.head["documents"]
                    or self.database.execute("SELECT COUNT(*) FROM shards").fetchone()[0] != self.head["sequence"]
                ):
                    raise ValueError("Rebuildable local cache has incomplete coverage; discard its index and retry")
                self._checked_miss_count = True
            return None
        self._verify(row["shard_id"])
        return {
            "cache_key": row["cache_key"],
            "ids_path": str((self.shards / row["shard_id"] / "ids.bin").relative_to(self.output)),
            "token_offset": row["token_offset"],
            "token_count": row["token_count"],
        }

    def encode_missing(self, tokenizer: Tokenizer, rows: Sequence[Mapping[str, Any]]) -> int:
        missing: dict[str, str] = {}
        for row in rows:
            if row["content_hash"] not in missing and self.lookup(row["content_hash"]) is None:
                missing[row["content_hash"]] = row["text"]
        if not missing:
            return 0
        if (
            self.head["documents"] + len(missing) > self.limits.max_documents
            or self.head["sequence"] + 1 > self.limits.max_shards
        ):
            raise ValueError("Token-cache partition admission limit reached; assign a new bounded partition")
        stage = _stage(self.root, ".incomplete-shard")
        try:
            encodings = _encode_batch(tokenizer, list(missing.values()))
            if len(encodings) != len(missing):
                raise ValueError("Tokenizer returned incomplete batch coverage")
            byte_count = sum(len(encoding.ids) * 4 for encoding in encodings)
            if self.head["token_bytes"] + byte_count > self.limits.max_token_bytes:
                raise ValueError("Token-cache partition byte limit reached")
            offsets = []
            offset = 0
            shard_hash = hashlib.sha256()
            with (stage / "ids.bin").open("xb") as binary:
                for content_hash, encoding in zip(missing, encodings):
                    ids = encoding.ids
                    data = _ids_bytes(ids)
                    binary.write(data)
                    offsets.append(
                        {
                            "content_hash": content_hash,
                            "cache_key": token_cache_key17(content_hash, self.tokenizer_sha256, self.policy),
                            "token_offset": offset,
                            "token_count": len(ids),
                            "ids_sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )
                    shard_hash.update(bytes.fromhex(content_hash))
                    offset += len(ids)
                binary.flush()
                os.fsync(binary.fileno())
            pq.write_table(pa.Table.from_pylist(offsets, schema=CACHE_SCHEMA), stage / "offsets.parquet", compression="zstd")
            _sync_file(stage / "offsets.parquet")
            shard_id = shard_hash.hexdigest()
            receipt = _sealed(
                stage / "receipt.json",
                {
                    "schema": "metis17.token-cache-shard/v1",
                    "identity": self.identity,
                    "shard_id": shard_id,
                    "records": len(offsets),
                    "token_count": offset,
                    "byte_count": offset * 4,
                    "ids_sha256": sha256_file(stage / "ids.bin"),
                    "offsets_sha256": sha256_file(stage / "offsets.parquet"),
                },
            )
            destination = self.shards / shard_id
            if destination.exists():
                raise ValueError("Cache publication would replace a committed shard")
            _sealed(self.root / "PENDING.json", {
                "identity": self.identity, "shard_id": shard_id,
                "shard_receipt_sha256": receipt["receipt_sha256"],
                "sequence": self.head["sequence"] + 1,
                "previous_commit_sha256": self.head["commit_sha256"],
                "documents": self.head["documents"] + len(missing),
                "token_bytes": self.head["token_bytes"] + byte_count,
            })
            _publish(stage, destination)
            self._recover()
            return len(missing)
        finally:
            if stage.exists() and not (self.root / "PENDING.json").exists():
                shutil.rmtree(stage)


def _encode_batch(tokenizer: Tokenizer, texts: list[str]) -> list[Any]:
    return tokenizer.encode_batch(texts, add_special_tokens=False)


def _verify_tokenization(
    output: Path, run: Path, identity: Mapping[str, Any], cache: _TokenCache
) -> dict[str, Any]:
    receipt = _read_sealed(run / "receipt.json")
    if receipt.get("schema") != "metis17.tokenization/v1" or receipt.get("identity") != identity:
        raise ValueError("Mismatched tokenization receipt")
    metadata_path = run / "offsets.parquet"
    if not metadata_path.is_file() or sha256_file(metadata_path) != receipt.get("metadata_sha256"):
        raise ValueError("Corrupt tokenization offset metadata")
    if pq.ParquetFile(metadata_path).schema_arrow != TOKENIZED_SCHEMA:
        raise ValueError("Corrupt tokenization offset schema")
    def expected_rows() -> Iterator[tuple[str, int, dict[str, Any]]]:
        for source in identity["inputs"]:
            source_row = 0
            for batch in _prepared_file(Path(source["path"])).iter_batches(
                batch_size=256, columns=[field.name for field in PREPARED_SCHEMA if field.name != "text"],
            ):
                for row in batch.to_pylist():
                    yield source["path"], source_row, row
                    source_row += 1
            if source_row != source["rows"]:
                raise ValueError("Prepared source has changed record coverage")

    expected = expected_rows()
    rows = tokens = 0
    for record in _parquet_rows(metadata_path):
        source = next(expected, None)
        if source is None or (record["source_shard"], record["source_row"]) != source[:2]:
            raise ValueError("Tokenization does not cover every source row exactly once")
        if any(record[key] != value for key, value in source[2].items()):
            raise ValueError("Tokenization record identity disagrees with its prepared source")
        cached = cache.lookup(record["content_hash"])
        if cached is None or any(record[key] != value for key, value in cached.items()):
            raise ValueError("Tokenization references inconsistent or missing cached IDs")
        rows += 1
        tokens += record["token_count"]
    if next(expected, None) is not None or rows != receipt.get("records") or tokens != receipt.get("token_count"):
        raise ValueError("Tokenization receipt has incomplete record or token coverage")
    if (
        receipt.get("metadata_path") != str(metadata_path.relative_to(output))
        or receipt.get("receipt_path") != str((run / "receipt.json").relative_to(output))
        or receipt.get("dtype") != "<u4"
        or receipt.get("offset_unit") != "tokens"
        or receipt.get("run_id") != digest_json(identity)
    ):
        raise ValueError("Tokenization receipt references the wrong metadata path")
    return receipt


def _tokenize_cached(
    input_paths: Sequence[Path], output: Path, tokenizer: Tokenizer,
    cache: _TokenCache, *, batch_size: int,
) -> dict[str, Any]:
    _positive_integer(batch_size, "batch_size")
    if len(input_paths) > cache.limits.max_input_paths:
        raise ValueError("Input shard count exceeds the bounded session limit")
    inputs = _input_snapshot(input_paths)
    identity = {
        "inputs": inputs,
        "tokenizer_sha256": cache.tokenizer_sha256,
        "tokenization_policy": cache.policy,
        "tokenization_policy_sha256": cache.policy_sha256,
        "cache_partition": cache.partition_id,
    }
    run_id = digest_json(identity)
    runs = output / "runs" / digest_json(cache.identity)
    runs.mkdir(parents=True, exist_ok=True)
    run = runs / run_id
    interrupted = runs / ".incomplete-run"
    if interrupted.exists():
        if interrupted.is_symlink() or not interrupted.is_dir():
            raise ValueError("Invalid interrupted tokenization staging directory")
        shutil.rmtree(interrupted)
    if run.exists():
        receipt = _verify_tokenization(output, run, identity, cache)
        for source in inputs:
            _unchanged(source)
        return receipt
    stage = _stage(runs, ".incomplete-run")
    try:
        records = total_tokens = encoded_documents = 0
        with pq.ParquetWriter(stage / "offsets.parquet", TOKENIZED_SCHEMA, compression="zstd") as writer:
            for source in inputs:
                source_row = 0
                parquet = _prepared_file(Path(source["path"]))
                for batch in parquet.iter_batches(batch_size=batch_size, columns=PREPARED_SCHEMA.names):
                    rows = batch.to_pylist()
                    for row in rows:
                        _checked_row(row)
                    encoded_documents += cache.encode_missing(tokenizer, rows)
                    metadata = []
                    for row in rows:
                        cached = cache.lookup(row["content_hash"])
                        if cached is None:
                            raise ValueError("Missing token IDs after encoding")
                        metadata.append({
                            "source_shard": source["path"], "source_row": source_row,
                            **{key: value for key, value in row.items() if key != "text"}, **cached,
                        })
                        source_row += 1
                        records += 1
                        total_tokens += cached["token_count"]
                    writer.write_table(pa.Table.from_pylist(metadata, schema=TOKENIZED_SCHEMA))
                if source_row != source["rows"]:
                    raise ValueError("Prepared source row count changed during tokenization")
                _unchanged(source)
        _sync_file(stage / "offsets.parquet")
        _sealed(stage / "receipt.json", {
            "schema": "metis17.tokenization/v1", "identity": identity,
            "created_at": utc_now(), "run_id": run_id,
            "records": records, "token_count": total_tokens,
            "encoded_documents": encoded_documents, "reused_documents": records - encoded_documents,
            "metadata_path": str((run / "offsets.parquet").relative_to(output)),
            "metadata_sha256": sha256_file(stage / "offsets.parquet"),
            "receipt_path": str((run / "receipt.json").relative_to(output)),
            "dtype": "<u4", "offset_unit": "tokens",
        })
        _publish(stage, run)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return _verify_tokenization(output, run, identity, cache)


class TokenizationSession17:
    """A reusable single-writer, bounded partition; SQLite never leaves local scratch.

    Reuse the same partition for retained text, and consume published offsets for
    count/pack/replay. Independent partitions may cache the same text separately;
    this is not a corpus-global tokenize-once guarantee. The caller owns partition
    assignment and aggregate disk budgets across workers.
    A scratch directory holds only one local index: switching partitions evicts
    the old derived index, never durable IDs. Give concurrent workers disjoint
    scratch directories. Reusing a session avoids reopening/recovering its index.
    """

    def __init__(
        self, output_dir: Path, tokenizer_dir: Path, *, scratch_dir: Path,
        partition_id: str, production: bool = True, limits: TokenCacheLimits17 | None = None,
    ) -> None:
        if not isinstance(partition_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", partition_id) is None:
            raise ValueError("partition_id must explicitly name a bounded worker/partition")
        self.output = Path(output_dir).resolve()
        self.tokenizer_dir = Path(tokenizer_dir)
        self.scratch_dir = Path(scratch_dir)
        self.partition_id = partition_id
        self.production = production
        self.limits = TokenCacheLimits17() if limits is None else limits
        self.limits.validate()
        self.cache: _TokenCache | None = None
        self._lock: Any = None
        self._scratch_lock: Any = None
        self._needs_recovery = False

    def __enter__(self) -> TokenizationSession17:
        if self.cache is not None:
            raise ValueError("Tokenization session is already open")
        scratch = _local_scratch(self.scratch_dir, self.output, self.limits.max_scratch_bytes)
        self.tokenizer, release = _load_artifact(self.tokenizer_dir, production=self.production)
        policy = tokenization_policy17()
        partition = (
            self.output / "cache" / release["tokenizer_sha256"] / digest_json(policy)
            / "partitions" / self.partition_id
        )
        self._scratch_lock = _locked(scratch / ".tokenizer-worker.lock", blocking=False)
        self._scratch_lock.__enter__()
        try:
            abandoned_sample = scratch / "tokenizer-samples"
            if abandoned_sample.exists():
                if abandoned_sample.is_symlink() or not abandoned_sample.is_dir():
                    raise ValueError("Invalid abandoned node-local sampling cache")
                shutil.rmtree(abandoned_sample)
            lock = _locked(partition / ".writer.lock", blocking=False)
            lock.__enter__()
            self._lock = lock
            self.cache = _TokenCache(
                self.output, release["tokenizer_sha256"], policy, scratch=scratch,
                partition_id=self.partition_id, limits=self.limits,
            )
        except Exception:
            if self._lock is not None:
                self._lock.__exit__(*sys.exc_info())
            self._lock = None
            self._scratch_lock.__exit__(*sys.exc_info())
            self._scratch_lock = None
            raise
        return self

    def tokenize_parquet(self, input_paths: Sequence[Path], *, batch_size: int = 256) -> dict[str, Any]:
        if self.cache is None:
            raise ValueError("Use TokenizationSession17 as a context manager")
        self.cache.seen_this_call.clear()
        try:
            if self._needs_recovery:
                self.cache._recover()
                self.cache._checked_miss_count = False
                self._needs_recovery = False
            return _tokenize_cached(input_paths, self.output, self.tokenizer, self.cache, batch_size=batch_size)
        except sqlite3.OperationalError as error:
            self._needs_recovery = True
            raise ValueError(f"Bounded node-local token-cache SQLite failed: {error}") from error
        except Exception:
            self._needs_recovery = True
            raise

    def __exit__(self, *exception: Any) -> None:
        try:
            if self.cache is not None:
                self.cache.close()
                self.cache = None
        finally:
            if self._lock is not None:
                self._lock.__exit__(*exception)
                self._lock = None
            if self._scratch_lock is not None:
                self._scratch_lock.__exit__(*exception)
                self._scratch_lock = None


def tokenize_parquet17(
    input_paths: Sequence[Path], output_dir: Path, tokenizer_dir: Path, *,
    scratch_dir: Path, partition_id: str, production: bool = True,
    batch_size: int = 256, limits: TokenCacheLimits17 | None = None,
) -> dict[str, Any]:
    """One-shot wrapper; reuse TokenizationSession17 for successive input chunks."""
    with TokenizationSession17(
        output_dir, tokenizer_dir, scratch_dir=scratch_dir, partition_id=partition_id,
        production=production, limits=limits,
    ) as session:
        return session.tokenize_parquet(input_paths, batch_size=batch_size)


def _sample_policy(
    stratum_byte_caps: Mapping[str, int],
    required_strata: Sequence[str] | None,
    stratum_columns: Sequence[str],
    minimum_bytes_per_stratum: Mapping[str, int] | None,
    seed: str,
) -> dict[str, Any]:
    caps = dict(stratum_byte_caps)
    if not caps or any(not isinstance(key, str) or not key for key in caps):
        raise ValueError("Supply explicit nonempty stratum names and byte caps")
    for cap in caps.values():
        _positive_integer(cap, "stratum byte cap")
    columns = list(stratum_columns)
    if not columns or len(set(columns)) != len(columns) or any(
        column not in {"source_id", "language", "category"} for column in columns
    ):
        raise ValueError("stratum_columns must select distinct category, language, or source_id columns")
    required = sorted(caps if required_strata is None else required_strata)
    if not required or len(set(required)) != len(required) or not set(required).issubset(caps):
        raise ValueError("required_strata must name distinct capped strata")
    minimums = {key: 1 for key in caps}
    if minimum_bytes_per_stratum is not None:
        if not set(minimum_bytes_per_stratum).issubset(caps):
            raise ValueError("Minimum byte thresholds must name capped strata")
        minimums.update(minimum_bytes_per_stratum)
    for key, minimum in minimums.items():
        _positive_integer(minimum, "minimum bytes per stratum")
        if minimum > caps[key]:
            raise ValueError("Minimum stratum bytes cannot exceed its byte cap")
    if not isinstance(seed, str) or not seed:
        raise ValueError("Sampling seed must be an explicit nonempty string")
    return {
        "algorithm": "sha256-ranked-whole-documents/v1",
        "stratum_columns": columns,
        "stratum_separator": "/",
        "stratum_byte_caps": caps,
        "minimum_bytes_per_stratum": minimums,
        "required_strata": required,
        "seed": seed,
        "exact_deduplication": "within-stratum-content-sha256",
        "oversized_document_policy": "skip-without-truncation",
    }


def _validate_sample(directory: Path, identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    receipt = _read_sealed(directory / "SAMPLE_RECEIPT.json")
    if receipt.get("schema") != "metis17.tokenizer-sample/v1":
        raise ValueError("Unsupported tokenizer sample receipt")
    if identity is not None and receipt.get("identity") != identity:
        raise ValueError("Tokenizer sample inputs or sampling policy changed; choose a new output directory")
    identity = receipt["identity"]
    policy = identity["sampling_policy"]
    if policy != _sample_policy(
        policy["stratum_byte_caps"], policy["required_strata"], policy["stratum_columns"],
        policy["minimum_bytes_per_stratum"], policy["seed"],
    ):
        raise ValueError("Invalid tokenizer sample policy")
    path = directory / "samples.parquet"
    if not path.is_file() or sha256_file(path) != receipt.get("samples_sha256"):
        raise ValueError("Corrupt tokenizer sample metadata")
    if pq.ParquetFile(path).schema_arrow != SAMPLE_SCHEMA:
        raise ValueError("Invalid tokenizer sample schema")
    selected = {key: {"documents": 0, "utf8_bytes": 0} for key in policy["stratum_byte_caps"]}
    previous = (-1, -1)
    for row in _parquet_rows(path):
        source_number, source_row = row["source_number"], row["source_row"]
        if (
            type(source_number) is not int
            or not 0 <= source_number < len(identity["inputs"])
            or type(source_row) is not int
            or (source_number, source_row) <= previous
        ):
            raise ValueError("Tokenizer sample has duplicate or unordered source references")
        source = identity["inputs"][source_number]
        stratum = row["stratum"]
        if (
            row["source_shard"] != source["path"]
            or not 0 <= source_row < source["rows"]
            or stratum not in selected
            or type(row["utf8_bytes"]) is not int
            or row["utf8_bytes"] <= 0
            or row["rank"] != digest_json(
                {"seed": policy["seed"], "stratum": stratum, "content_hash": _digest(row["content_hash"], "sample content_hash")}
            )
        ):
            raise ValueError("Invalid tokenizer sample reference or rank")
        selected[stratum]["documents"] += 1
        selected[stratum]["utf8_bytes"] += row["utf8_bytes"]
        previous = (source_number, source_row)
    missing = []
    for stratum, stats in selected.items():
        coverage = receipt["coverage"][stratum]
        if (
            stats["utf8_bytes"] > policy["stratum_byte_caps"][stratum]
            or coverage["selected_documents"] != stats["documents"]
            or coverage["selected_bytes"] != stats["utf8_bytes"]
        ):
            raise ValueError("Tokenizer sample coverage or byte cap mismatch")
        if stratum in policy["required_strata"] and stats["utf8_bytes"] < policy["minimum_bytes_per_stratum"][stratum]:
            missing.append(stratum)
    missing.sort()
    if (
        receipt.get("missing_strata") != missing
        or receipt.get("ready") is not (not missing)
        or receipt.get("selected_documents") != sum(stats["documents"] for stats in selected.values())
        or receipt.get("selected_bytes") != sum(stats["utf8_bytes"] for stats in selected.values())
    ):
        raise ValueError("Tokenizer sample readiness does not match actual coverage")
    return receipt


def build_tokenizer_sample17(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    scratch_dir: Path,
    stratum_byte_caps: Mapping[str, int],
    required_strata: Sequence[str] | None = None,
    stratum_columns: Sequence[str] = ("category",),
    minimum_bytes_per_stratum: Mapping[str, int] | None = None,
    seed: str = "metis1.7-tokenizer-v1",
    batch_size: int = 256,
    max_scratch_bytes: int = 512 * 1024**2,
    max_candidate_documents: int = 1_000_000,
    max_input_paths: int = 4_096,
) -> dict[str, Any]:
    """Seal a deterministic sample of eligible prepared rows, without copying text.

    Caps/minimums count exact UTF-8 bytes of whole documents. Keys join the named
    stratum_columns with '/'. A missing required stratum produces ready=False;
    iter_tokenizer_sample17 refuses that sample unless explicitly in test mode.
    Candidate ranking uses bounded node-local SQLite. The first pass reads only
    metadata; only possibly fitting candidates' row groups have text decoded.
    Coverage deliberately leaves eligible_bytes unknown rather than scanning all
    text to count it. Exact selected bytes and every source-row count are reported.
    """
    _positive_integer(batch_size, "batch_size")
    _positive_integer(max_candidate_documents, "max_candidate_documents")
    _positive_integer(max_input_paths, "max_input_paths")
    if len(input_paths) > max_input_paths:
        raise ValueError("Sampling input shard count exceeds its explicit bound")
    policy = _sample_policy(stratum_byte_caps, required_strata, stratum_columns, minimum_bytes_per_stratum, seed)
    output = Path(output_dir).resolve()
    scratch = _local_scratch(scratch_dir, output, max_scratch_bytes)
    with _locked(scratch / ".tokenizer-worker.lock", blocking=False), _locked(output.parent / f".{output.name}.sampling.lock"):
        inputs = _input_snapshot(input_paths)
        identity = {"inputs": inputs, "sampling_policy": policy}
        if output.exists() and any(output.iterdir()):
            result = _validate_sample(output, identity)
            for source in inputs:
                _unchanged(source)
            return result
        stage = _stage(output.parent, f".{output.name}.sampling-incomplete")
        old_index = scratch / "token-cache"
        if old_index.exists():
            if old_index.is_symlink() or not old_index.is_dir():
                raise ValueError("Invalid node-local cache directory")
            shutil.rmtree(old_index)
        local = _stage(scratch, "tokenizer-samples")
        database = _local_database(local / "candidates.sqlite3", max_scratch_bytes)
        try:
            # Both traversal orders have persistent indexes; no corpus-sized sort is needed.
            database.executescript(
                """
                CREATE TABLE candidates (
                    stratum TEXT NOT NULL, rank TEXT NOT NULL, content_hash TEXT NOT NULL,
                    source_number INTEGER NOT NULL, source_row INTEGER NOT NULL,
                    doc_id TEXT NOT NULL, character_count INTEGER NOT NULL,
                    PRIMARY KEY (stratum, rank, content_hash), UNIQUE (stratum, content_hash)
                ) WITHOUT ROWID;
                CREATE TABLE selected (
                    source_number INTEGER NOT NULL, source_row INTEGER NOT NULL,
                    content_hash TEXT NOT NULL, doc_id TEXT NOT NULL,
                    stratum TEXT NOT NULL, rank TEXT NOT NULL, utf8_bytes INTEGER NOT NULL,
                    PRIMARY KEY (source_number, source_row)
                ) WITHOUT ROWID;
                """
            )
            coverage = {
                key: {
                    "byte_cap": cap,
                    "minimum_bytes": policy["minimum_bytes_per_stratum"][key],
                    "eligible_documents": 0,
                    "eligible_bytes": None,
                    "eligible_characters": 0,
                    "unique_candidates": 0,
                    "duplicate_documents": 0,
                    "oversized_documents": 0,
                    "oversized_documents_complete": False,
                    "empty_documents": 0,
                    "selected_documents": 0,
                    "selected_bytes": 0,
                    "decoded_documents": 0,
                    "decoded_bytes": 0,
                }
                for key, cap in policy["stratum_byte_caps"].items()
            }
            uncapped_documents = total_records = candidate_count = 0
            for source_number, source in enumerate(inputs):
                source_row = 0
                for batch in _prepared_file(Path(source["path"])).iter_batches(
                    batch_size=batch_size, columns=[field.name for field in PREPARED_SCHEMA if field.name != "text"],
                ):
                    with database:
                        for row in batch.to_pylist():
                            _checked_metadata(row)
                            characters = row["character_count"]
                            stratum = "/".join(row[column] for column in policy["stratum_columns"])
                            if stratum not in coverage:
                                uncapped_documents += 1
                            else:
                                stats = coverage[stratum]
                                stats["eligible_documents"] += 1
                                stats["eligible_characters"] += characters
                                if characters == 0:
                                    stats["empty_documents"] += 1
                                elif characters > stats["byte_cap"]:
                                    stats["oversized_documents"] += 1
                                else:
                                    rank = digest_json({"seed": seed, "stratum": stratum, "content_hash": row["content_hash"]})
                                    cursor = database.execute(
                                        "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
                                        (stratum, rank, row["content_hash"], source_number, source_row, row["doc_id"], characters),
                                    )
                                    stats["unique_candidates" if cursor.rowcount else "duplicate_documents"] += 1
                                    candidate_count += cursor.rowcount
                                    if candidate_count > max_candidate_documents:
                                        raise ValueError("Tokenizer sample candidate count exceeds its explicit local scratch bound")
                            source_row += 1
                            total_records += 1
                if source_row != source["rows"]:
                    raise ValueError("Prepared row coverage changed during sampling")
                _unchanged(source)
            with database:
                for stratum, stats in coverage.items():
                    candidates = database.execute(
                        "SELECT * FROM candidates WHERE stratum=? ORDER BY rank, content_hash", (stratum,)
                    )
                    while block := candidates.fetchmany(batch_size):
                        fitting = [
                            row for row in block
                            if row["character_count"] <= stats["byte_cap"] - stats["selected_bytes"]
                        ]
                        sizes = _candidate_sizes(inputs, fitting, batch_size)
                        stats["decoded_documents"] += len(fitting)
                        stats["decoded_bytes"] += sum(sizes.values())
                        for candidate in block:
                            byte_count = sizes.get(candidate["content_hash"])
                            if byte_count is not None and byte_count > stats["byte_cap"]:
                                stats["oversized_documents"] += 1
                            if byte_count is not None and stats["selected_bytes"] + byte_count <= stats["byte_cap"]:
                                database.execute(
                                    "INSERT INTO selected VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (*tuple(candidate[key] for key in (
                                        "source_number", "source_row", "content_hash", "doc_id", "stratum", "rank",
                                    )), byte_count),
                                )
                                stats["selected_bytes"] += byte_count
                                stats["selected_documents"] += 1
            with pq.ParquetWriter(stage / "samples.parquet", SAMPLE_SCHEMA, compression="zstd") as writer:
                cursor = database.execute("SELECT * FROM selected ORDER BY source_number, source_row")
                while candidates := cursor.fetchmany(batch_size):
                    rows = [
                        {**dict(row), "source_shard": inputs[row["source_number"]]["path"]}
                        for row in candidates
                    ]
                    writer.write_table(pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA))
            database.close()
            shutil.rmtree(local)
            _sync_file(stage / "samples.parquet")
            missing = sorted(
                key for key in policy["required_strata"]
                if coverage[key]["selected_bytes"] < policy["minimum_bytes_per_stratum"][key]
            )
            _sealed(
                stage / "SAMPLE_RECEIPT.json",
                {
                    "schema": "metis17.tokenizer-sample/v1",
                    "created_at": utc_now(),
                    "identity": identity,
                    "records_scanned": total_records,
                    "candidate_documents": candidate_count,
                    "scratch_limits": {
                        "max_scratch_bytes": max_scratch_bytes,
                        "max_candidate_documents": max_candidate_documents,
                        "max_input_paths": max_input_paths,
                    },
                    "uncapped_documents": uncapped_documents,
                    "coverage": coverage,
                    "missing_strata": missing,
                    "ready": not missing,
                    "selected_documents": sum(stats["selected_documents"] for stats in coverage.values()),
                    "selected_bytes": sum(stats["selected_bytes"] for stats in coverage.values()),
                    "samples_path": "samples.parquet",
                    "samples_sha256": sha256_file(stage / "samples.parquet"),
                },
            )
            _validate_sample(stage, identity)
            _publish(stage, output)
            return _validate_sample(output, identity)
        except sqlite3.OperationalError as error:
            raise ValueError(f"Bounded node-local sampling SQLite failed: {error}") from error
        finally:
            database.close()
            if local.exists():
                shutil.rmtree(local)
            if stage.exists():
                shutil.rmtree(stage)


def iter_tokenizer_sample17(
    directory: Path, *, production: bool = True, batch_size: int = 256
) -> Iterator[str]:
    """Read only selected prepared row groups and validate source hashes before/after."""
    _positive_integer(batch_size, "batch_size")
    if type(production) is not bool:
        raise ValueError("production must be a boolean")
    directory = Path(directory).resolve()
    receipt = _validate_sample(directory)
    if production and not receipt["ready"]:
        raise ValueError(f"Tokenizer sample is not ready; missing strata: {receipt['missing_strata']}")
    references = iter(_parquet_rows(directory / "samples.parquet", batch_size=batch_size))
    pending = next(references, None)
    yielded = 0
    for source_number, source in enumerate(receipt["identity"]["inputs"]):
        _unchanged(source)
        parquet = _prepared_file(Path(source["path"]))
        group_start = 0
        for group_number in range(parquet.metadata.num_row_groups):
            group_end = group_start + parquet.metadata.row_group(group_number).num_rows
            if pending is not None and pending["source_number"] == source_number and pending["source_row"] < group_end:
                source_row = group_start
                for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group_number], columns=PREPARED_SCHEMA.names):
                    for row in batch.to_pylist():
                        if pending is not None and pending["source_number"] == source_number and pending["source_row"] == source_row:
                            byte_count = _checked_row(row)
                            stratum = "/".join(row[column] for column in receipt["identity"]["sampling_policy"]["stratum_columns"])
                            if (
                                pending["content_hash"] != row["content_hash"]
                                or pending["doc_id"] != row["doc_id"]
                                or pending["utf8_bytes"] != byte_count
                                or pending["stratum"] != stratum
                            ):
                                raise ValueError("Sample reference does not match its prepared record")
                            yield row["text"]
                            yielded += 1
                            pending = next(references, None)
                        source_row += 1
                    if pending is None or pending["source_number"] != source_number or pending["source_row"] >= group_end:
                        break
            group_start = group_end
        _unchanged(source)
    if pending is not None or yielded != receipt["selected_documents"]:
        raise ValueError("Incomplete tokenizer sample coverage")
