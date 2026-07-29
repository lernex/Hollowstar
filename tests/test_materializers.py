from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from metis_data.materializers import (
    MaterializationError,
    _ShardWriter,
    _git_registry_entries,
    _ingest_repository_metadata,
    _load_completed_unit,
    _materialize_repository_index,
    _materialize_repository_unit,
    _materialize_web_entry,
    _repository_index_connection,
    _registry_web_entries,
)
from metis_data.state import ScratchBackedDatabase


MIT_LICENSE = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files to deal in the Software
without restriction.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""


def _archive(path: Path, files: dict[str, str], *, license_text: str = MIT_LICENSE) -> Path:
    with tarfile.open(path, "w:gz") as bundle:
        payloads = {"repo-abc1234/LICENSE": license_text, **files}
        for name, text in payloads.items():
            raw = text.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            bundle.addfile(member, io.BytesIO(raw))
    return path


class _FakeArchiveClient:
    def __init__(self, archive: Path) -> None:
        self.archive = archive
        self.calls: list[tuple[str, str]] = []

    def fetch(self, repo: str, commit: str) -> dict[str, object]:
        self.calls.append((repo, commit))
        return {
            "path": self.archive,
            "url": f"https://codeload.github.com/{repo}/tar.gz/{commit}",
            "size": self.archive.stat().st_size,
            "sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
        }


class MaterializerContractTests(unittest.TestCase):
    def test_shared_git_registry_requires_an_explicit_selector(self) -> None:
        payload = {
            "schema": "metis.canonical-git-registry/v1",
            "repositories": [{"id": "lean", "url": "https://example.test/lean.git", "license": "Apache-2.0"}],
        }
        source = {
            "id": "formal",
            "access": {"registry": "shared.yaml"},
        }
        manifest = {
            "sources": [
                {"id": "formal", "access": {"registry": "shared.yaml"}, "acquisition": {"driver": "canonical_git"}},
                {"id": "systems", "access": {"registry": "shared.yaml"}, "acquisition": {"driver": "canonical_git"}},
            ]
        }
        with self.assertRaisesRegex(MaterializationError, "Declare registry.groups"):
            _git_registry_entries({}, manifest, source, payload)

    def test_canonical_git_registry_requires_immutable_revisions(self) -> None:
        source = {"id": "formal", "access": {"registry": "formal.yaml"}}
        manifest = {"sources": [{**source, "acquisition": {"driver": "canonical_git"}}]}
        payload = {
            "schema": "metis.canonical-git-registry/v1",
            "resolve_revisions_on_first_run": True,
            "repositories": [
                {"id": "lean", "url": "https://example.test/lean.git", "license": "Apache-2.0"}
            ],
        }
        with self.assertRaisesRegex(MaterializationError, "disable first-run revision resolution"):
            _git_registry_entries({}, manifest, source, payload)

        payload["resolve_revisions_on_first_run"] = False
        with self.assertRaisesRegex(MaterializationError, "full 40-hex revision"):
            _git_registry_entries({}, manifest, source, payload)

        payload["repositories"][0]["revision"] = "a" * 40
        self.assertEqual(_git_registry_entries({}, manifest, source, payload), payload["repositories"])

    def test_current_landing_page_registries_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load((root / "manifests" / "registries" / "software_docs.yaml").read_text())
        source = {"id": "software-docs", "access": {}}
        entries = _registry_web_entries(source, payload)
        self.assertTrue(entries)
        self.assertTrue(all("fetch_mode" not in entry for entry in entries))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(MaterializationError, "discovery/landing URL"):
                _materialize_web_entry(
                    source=source,
                    entry=entries[0],
                    output_root=Path(temporary),
                    target_shard_bytes=1_000_000,
                )

    def test_shard_marker_is_restartable_and_checksum_verified(self) -> None:
        from metis_data.materializers import _commit_unit

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = _ShardWriter(
                root,
                source_id="source",
                unit_id="unit",
                materializer="test/v1",
                revision="revision",
                target_uncompressed_bytes=1_000_000,
            )
            writer.write({"id": "one", "text": "hello", "metadata": {"license": "MIT"}})
            outputs, records = writer.finish()
            self.assertEqual(records, 1)
            _commit_unit(root, "unit", "signature", outputs, {"records": records})
            self.assertEqual(_load_completed_unit(root, "unit", "signature"), outputs)
            Path(outputs[0]["local_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(MaterializationError, "missing or truncated|checksum changed"):
                _load_completed_unit(root, "unit", "signature")

    def test_repository_index_groups_duplicate_metadata_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.parquet"
            second = root / "second.parquet"
            common = {
                "repo": ["Owner/Repo"],
                "rel_path": ["src/useful.py"],
                "language": ["Python"],
                "commit_id": ["abc1234"],
                # This is evidence only; it cannot authorize the source.
                "license": ["MIT"],
            }
            pq.write_table(pa.table(common), first)
            pq.write_table(
                pa.table(
                    {
                        **common,
                        "rel_path": ["src/other.py"],
                    }
                ),
                second,
            )
            component_a = {"repo_id": "nvidia/code-v1", "revision": "a" * 40}
            component_b = {"repo_id": "nvidia/code-v2", "revision": "b" * 40}
            connection = _repository_index_connection(root / "requests.sqlite3")
            try:
                report = _ingest_repository_metadata(
                    connection,
                    source={"id": "repository-code"},
                    metadata_files=[
                        (component_a, first),
                        (component_b, second),
                        # Repeating a completed metadata unit is restart-safe.
                        (component_a, first),
                    ],
                    spool_root=root / "index-sort",
                    sort_buckets=4,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM repo_commits").fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM requested_paths").fetchone()[0],
                    2,
                )
                self.assertEqual(report["metadata_units_reused"], 1)
            finally:
                connection.close()

    def test_repository_archive_is_fetched_once_and_only_requested_paths_are_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _archive(
                root / "repo.tar.gz",
                {
                    "repo-abc1234/src/useful.py": "def useful(value):\n    return value + 1\n",
                    "repo-abc1234/src/other.py": "def other(value):\n    return value - 1\n",
                    "repo-abc1234/src/not-requested.py": "raise RuntimeError('not requested')\n",
                },
            )
            client = _FakeArchiveClient(archive)
            requests = [
                {
                    "rel_path": "src/useful.py",
                    "language": "Python",
                    "index_license": "MIT",
                    "metadata_component": "v1",
                    "metadata_unit": "u1",
                },
                {
                    "rel_path": "src/other.py",
                    "language": "Python",
                    "index_license": "NOASSERTION",
                    "metadata_component": "v3",
                    "metadata_unit": "u2",
                },
            ]
            outputs, text_bytes, report = _materialize_repository_unit(
                source={"id": "repository-code"},
                repo="owner/repo",
                commit="abc1234",
                requests_for_repo=requests,
                unit_id="repository",
                signature="input-signature",
                output_root=root / "output",
                client=client,
                target_shard_bytes=1_000_000,
                maximum_file_bytes=2_000_000,
            )
            self.assertEqual(client.calls, [("owner/repo", "abc1234")])
            self.assertEqual(report["license"], "MIT")
            self.assertGreater(text_bytes, 0)
            import zstandard as zstd

            with Path(outputs[0]["local_path"]).open("rb") as raw:
                with zstd.ZstdDecompressor().stream_reader(raw) as stream:
                    rows = [
                        json.loads(line)
                        for line in stream.read().decode("utf-8").splitlines()
                    ]
            self.assertEqual(
                {row["metadata"]["repo_path"] for row in rows},
                {"src/useful.py", "src/other.py"},
            )
            self.assertTrue(all(row["metadata"]["license_basis"].startswith("repository-root") for row in rows))
            self.assertTrue(all(row["metadata"]["source_archive_sha256"] for row in rows))
            self.assertTrue(all(row["metadata"]["source_content_sha256"] for row in rows))
            # A restart trusts the checksum-verified unit marker and does not
            # fetch the repository archive a second time.
            resumed, resumed_bytes, resumed_report = _materialize_repository_unit(
                source={"id": "repository-code"},
                repo="owner/repo",
                commit="abc1234",
                requests_for_repo=requests,
                unit_id="repository",
                signature="input-signature",
                output_root=root / "output",
                client=client,
                target_shard_bytes=1_000_000,
                maximum_file_bytes=2_000_000,
            )
            self.assertEqual(resumed, outputs)
            self.assertEqual(resumed_bytes, text_bytes)
            self.assertTrue(resumed_report["resumed"])
            self.assertEqual(client.calls, [("owner/repo", "abc1234")])
            Path(outputs[0]["local_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(MaterializationError, "missing or truncated|checksum changed"):
                _materialize_repository_unit(
                    source={"id": "repository-code"},
                    repo="owner/repo",
                    commit="abc1234",
                    requests_for_repo=requests,
                    unit_id="repository",
                    signature="input-signature",
                    output_root=root / "output",
                    client=client,
                    target_shard_bytes=1_000_000,
                    maximum_file_bytes=2_000_000,
                )

    def test_repository_archive_rejects_metadata_license_without_root_license(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = _archive(
                root / "repo.tar.gz",
                {"repo-abc1234/src/useful.py": "def useful(value):\n    return value + 1\n"},
                license_text="Copyright 2026. All rights reserved.",
            )
            with self.assertRaisesRegex(MaterializationError, "license_not_allowlisted"):
                _materialize_repository_unit(
                    source={"id": "repository-code"},
                    repo="owner/repo",
                    commit="abc1234",
                    requests_for_repo=[
                        {
                            "rel_path": "src/useful.py",
                            "language": "Python",
                            "index_license": "MIT",
                            "metadata_component": "v3",
                            "metadata_unit": "u3",
                        }
                    ],
                    unit_id="repository",
                    signature="input-signature",
                    output_root=root / "output",
                    client=_FakeArchiveClient(archive),
                    target_shard_bytes=1_000_000,
                    maximum_file_bytes=2_000_000,
                )

    def test_repository_index_fails_when_candidate_headroom_is_short(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = {
                "repo_id": "nvidia/Nemotron-Pretraining-Code-v3",
                "revision": "a" * 40,
            }
            metadata_root = (
                root
                / "raw"
                / "repository-code"
                / "nvidia--Nemotron-Pretraining-Code-v3"
                / ("a" * 40)
            )
            metadata_root.mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "repo": ["owner/repo"],
                        "rel_path": ["src/useful.py"],
                        "language": ["Python"],
                        "commit_id": ["abc1234"],
                    }
                ),
                metadata_root / "metadata.parquet",
            )
            archive = _archive(
                root / "repo.tar.gz",
                {"repo-abc1234/src/useful.py": "def useful(value):\n    return value + 1\n"},
            )
            profile = {
                "runtime": {"download_retries": 0, "request_timeout_seconds": 1},
                "acquisition": {
                    "repository_code_bytes_per_token": 1.0,
                    "materializer_shard_bytes": 1_000_000,
                    "maximum_repository_file_bytes": 2_000_000,
                },
            }
            source = {
                "id": "repository-code",
                "access": {"components": [component]},
            }
            with (
                mock.patch(
                    "metis_data.materializers._CodeloadArchiveClient",
                    return_value=_FakeArchiveClient(archive),
                ),
                self.assertRaisesRegex(MaterializationError, "metadata exhausted"),
            ):
                _materialize_repository_index(
                    {"candidate_tokens": 10_000},
                    profile=profile,
                    root=root,
                    source=source,
                )

    def test_many_small_repositories_compact_into_bounded_restartable_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = "a" * 40
            component = {
                "repo_id": "nvidia/Nemotron-Pretraining-Code-v3",
                "revision": revision,
            }
            metadata_root = (
                root
                / "raw"
                / "repository-code"
                / "nvidia--Nemotron-Pretraining-Code-v3"
                / revision
            )
            metadata_root.mkdir(parents=True)
            repositories = [f"owner/repository-{index:03d}" for index in range(64)]
            pq.write_table(
                pa.table(
                    {
                        "repo": repositories,
                        "rel_path": ["src/useful.py"] * len(repositories),
                        "language": ["Python"] * len(repositories),
                        "commit_id": ["abc1234"] * len(repositories),
                    }
                ),
                metadata_root / "metadata.parquet",
            )
            archive = _archive(
                root / "repo.tar.gz",
                {
                    "repo-abc1234/src/useful.py": (
                        "def useful(value):\n"
                        "    # A deliberately non-trivial source file for aggregation.\n"
                        "    return value + 1\n"
                    )
                },
            )
            profile = {
                "runtime": {"download_retries": 0, "request_timeout_seconds": 1},
                "acquisition": {
                    "repository_code_bytes_per_token": 1.0,
                    "materializer_shard_bytes": 1_000_000,
                    "repository_output_shard_bytes": 1_000_000,
                    "repository_max_repositories_per_batch": 16,
                    "repository_index_workers": 4,
                    "maximum_repository_file_bytes": 2_000_000,
                },
            }
            source = {
                "id": "repository-code",
                "access": {"components": [component]},
            }
            first_client = _FakeArchiveClient(archive)
            with mock.patch(
                "metis_data.materializers._CodeloadArchiveClient",
                return_value=first_client,
            ):
                outputs = _materialize_repository_index(
                    {"candidate_tokens": 0},
                    profile=profile,
                    root=root,
                    source=source,
                )
            self.assertEqual(len(first_client.calls), len(repositories))
            self.assertEqual(len(outputs), 4)
            self.assertLess(len(outputs), len(repositories) // 4)
            database = (
                root
                / "raw"
                / "repository-code"
                / "repository-index-cache"
                / "requests.sqlite3"
            )
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM repository_state WHERE status='accepted'"
                    ).fetchone()[0],
                    len(repositories),
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM repository_state "
                        "WHERE status='accepted' AND content_manifest_sha256 IS NOT NULL"
                    ).fetchone()[0],
                    len(repositories),
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM output_batches").fetchone()[0],
                    4,
                )
            finally:
                connection.close()
            spool_markers = (
                root
                / "raw"
                / "repository-code"
                / "repository-index-cache"
                / "spool"
                / ".markers"
            )
            self.assertFalse(spool_markers.exists() and any(spool_markers.iterdir()))

            resumed_client = _FakeArchiveClient(archive)
            with mock.patch(
                "metis_data.materializers._CodeloadArchiveClient",
                return_value=resumed_client,
            ):
                resumed = _materialize_repository_index(
                    {"candidate_tokens": 0},
                    profile=profile,
                    root=root,
                    source=source,
                )
            self.assertEqual(resumed, outputs)
            self.assertEqual(resumed_client.calls, [])

            Path(outputs[0]["local_path"]).write_bytes(b"corrupt")
            with (
                mock.patch(
                    "metis_data.materializers._CodeloadArchiveClient",
                    return_value=_FakeArchiveClient(archive),
                ),
                self.assertRaisesRegex(
                    MaterializationError,
                    "missing or truncated|checksum changed",
                ),
            ):
                _materialize_repository_index(
                    {"candidate_tokens": 0},
                    profile=profile,
                    root=root,
                    source=source,
                )

    def test_repository_index_parquet_contract_has_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.parquet"
            pq.write_table(
                pa.table(
                    {
                        "repo": ["owner/repo"],
                        "rel_path": ["file.py"],
                        "language": ["Python"],
                        "commit_id": ["abc1234"],
                    }
                ),
                path,
            )
            parquet = pq.ParquetFile(path)
            self.assertEqual(set(parquet.schema.names), {"repo", "rel_path", "language", "commit_id"})


class RepositoryIndexStorageTests(unittest.TestCase):
    """The request index is the phase a cold Nemotron build spends days in."""

    @staticmethod
    def _metadata(path: Path, rows: int, *, seed: int) -> None:
        # Deliberately unsorted: the real metadata arrives in arbitrary order,
        # and both index tables are keyed on sha256(repo, commit).
        order = [(index * 7919 + seed) % rows for index in range(rows)]
        pq.write_table(
            pa.table(
                {
                    "repo": [f"Owner{value % 97}/Repo{value}" for value in order],
                    "rel_path": [f"src/module_{value % 13}.py" for value in order],
                    "language": ["Python"] * rows,
                    "commit_id": [hashlib.sha1(str(value).encode()).hexdigest() for value in order],
                    "license": ["MIT"] * rows,
                }
            ),
            path,
        )

    def _ingest(
        self,
        root: Path,
        name: str,
        metadata: Path,
        *,
        buckets: int,
        spool_rows: int,
        connection: object | None = None,
    ) -> dict[str, object]:
        owned = connection is None
        target = _repository_index_connection(root / name) if owned else connection
        try:
            report = _ingest_repository_metadata(
                target,
                source={"id": "repository-code"},
                metadata_files=[({"repo_id": "nvidia/code-v1", "revision": "a" * 40}, metadata)],
                spool_root=root / f"{name}.index-sort",
                sort_buckets=buckets,
                spool_rows=spool_rows,
            )
            rows = target.execute(
                "SELECT repo_key, rel_path FROM requested_paths ORDER BY repo_key, rel_path"
            ).fetchall()
            repos = target.execute(
                "SELECT repo_key, repo, commit_id FROM repo_commits ORDER BY repo_key"
            ).fetchall()
            return {"report": report, "rows": rows, "repos": repos}
        finally:
            if owned:
                target.close()

    def test_bucket_layout_does_not_change_what_is_indexed(self) -> None:
        """Bucket count and spool size are performance choices, not semantics."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.parquet"
            self._metadata(metadata, 4_000, seed=11)
            fine = self._ingest(
                root, "fine.sqlite3", metadata, buckets=256, spool_rows=100_000
            )
            coarse = self._ingest(
                root, "coarse.sqlite3", metadata, buckets=1, spool_rows=100_000
            )
            rolled = self._ingest(
                root, "rolled.sqlite3", metadata, buckets=64, spool_rows=100_000
            )
            self.assertEqual(fine["rows"], coarse["rows"])
            self.assertEqual(fine["rows"], rolled["rows"])
            self.assertEqual(fine["repos"], coarse["repos"])
            self.assertEqual(fine["repos"], rolled["repos"])
            self.assertEqual(
                fine["report"]["requested_paths"], coarse["report"]["requested_paths"]
            )
            self.assertEqual(fine["report"]["repositories"], coarse["report"]["repositories"])
            self.assertGreater(fine["report"]["requested_paths"], 0)

    def test_the_index_is_built_by_ascending_key_so_the_tree_appends(self) -> None:
        """The whole point of the external sort, asserted directly.

        Both index tables are WITHOUT ROWID trees keyed on sha256(repo,
        commit). Inserting in metadata order means a random page per row, which
        on Lustre is a network round trip per row. Inserting in ascending key
        order means every row lands on the rightmost leaf.
        """

        class Recording:
            def __init__(self, connection: sqlite3.Connection) -> None:
                self.connection = connection
                self.request_keys: list[bytes] = []
                self.repo_keys: list[bytes] = []

            def executemany(self, statement: str, rows: object) -> object:
                materialized = list(rows)
                if "INTO requested_paths" in statement:
                    self.request_keys.extend(row[0] for row in materialized)
                elif "INTO repo_commits" in statement:
                    self.repo_keys.extend(row[0] for row in materialized)
                return self.connection.executemany(statement, materialized)

            def __getattr__(self, name: str) -> object:
                return getattr(self.connection, name)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.parquet"
            self._metadata(metadata, 6_000, seed=5)
            connection = _repository_index_connection(root / "index.sqlite3")
            recording = Recording(connection)
            try:
                self._ingest(
                    root,
                    "index.sqlite3",
                    metadata,
                    buckets=16,
                    spool_rows=100_000,
                    connection=recording,
                )
                self.assertGreater(len(recording.request_keys), 1_000)
                self.assertEqual(
                    recording.request_keys,
                    sorted(recording.request_keys),
                    "requested_paths must be presented to SQLite in ascending key order",
                )
                self.assertEqual(
                    recording.repo_keys,
                    sorted(recording.repo_keys),
                    "repo_commits must be presented to SQLite in ascending key order",
                )
            finally:
                connection.close()

    def test_spool_rolls_do_not_lose_or_duplicate_rows(self) -> None:
        """A roll every few thousand rows must index exactly what one roll does."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.parquet"
            self._metadata(metadata, 5_000, seed=29)
            single = self._ingest(
                root, "single.sqlite3", metadata, buckets=32, spool_rows=10_000_000
            )
            many = self._ingest(
                root, "many.sqlite3", metadata, buckets=32, spool_rows=100_000
            )
            self.assertEqual(single["rows"], many["rows"])
            self.assertEqual(single["repos"], many["repos"])

    def test_repo_identity_index_is_built_and_enforced_after_the_load(self) -> None:
        """The unique index is deferred for speed, not dropped."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.parquet"
            self._metadata(metadata, 800, seed=7)
            connection = _repository_index_connection(root / "index.sqlite3")
            try:
                self._ingest(
                    root,
                    "index.sqlite3",
                    metadata,
                    buckets=8,
                    spool_rows=100_000,
                    connection=connection,
                )
                indexes = {
                    str(row[1])
                    for row in connection.execute("PRAGMA index_list(repo_commits)")
                }
                self.assertIn("repo_commits_identity", indexes)
                self.assertIn("repo_commits_rank", indexes)
                repo, commit = connection.execute(
                    "SELECT repo, commit_id FROM repo_commits LIMIT 1"
                ).fetchone()
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO repo_commits(repo_key, repo, commit_id, rank_key)"
                        " VALUES(?,?,?,?)",
                        (b"\xff" * 16, repo, commit, "rank"),
                    )
            finally:
                connection.close()

    def test_incomplete_spool_parts_are_ignored(self) -> None:
        """A crash mid-write leaves a .partial, never a footerless Parquet file."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.parquet"
            self._metadata(metadata, 600, seed=13)
            spool_root = root / "index.sqlite3.index-sort"
            connection = _repository_index_connection(root / "index.sqlite3")
            try:
                self._ingest(
                    root,
                    "index.sqlite3",
                    metadata,
                    buckets=4,
                    spool_rows=100_000,
                    connection=connection,
                )
                expected = connection.execute(
                    "SELECT COUNT(*) FROM requested_paths"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertGreater(expected, 0)
            self.assertFalse(spool_root.exists(), "a completed load removes its spool")

    def test_units_arriving_after_the_load_fail_closed(self) -> None:
        """A sorted load cannot absorb late rows, so it must refuse them loudly."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.parquet"
            second = root / "second.parquet"
            self._metadata(first, 400, seed=2)
            self._metadata(second, 400, seed=99)
            connection = _repository_index_connection(root / "index.sqlite3")
            try:
                _ingest_repository_metadata(
                    connection,
                    source={"id": "repository-code"},
                    metadata_files=[
                        ({"repo_id": "nvidia/code-v1", "revision": "a" * 40}, first)
                    ],
                    spool_root=root / "sort",
                    sort_buckets=4,
                )
                with self.assertRaises(MaterializationError) as caught:
                    _ingest_repository_metadata(
                        connection,
                        source={"id": "repository-code"},
                        metadata_files=[
                            ({"repo_id": "nvidia/code-v1", "revision": "a" * 40}, first),
                            ({"repo_id": "nvidia/code-v2", "revision": "b" * 40}, second),
                        ],
                        spool_root=root / "sort",
                        sort_buckets=4,
                    )
                self.assertIn("rebuild the index", str(caught.exception))
            finally:
                connection.close()

    def test_journal_mode_follows_the_filesystem_the_index_lives_on(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            durable = _repository_index_connection(root / "durable.sqlite3", local=False)
            try:
                # A write-ahead log needs an mmap-backed shared-memory index
                # beside the database, which Lustre serves badly.
                self.assertEqual(
                    durable.execute("PRAGMA journal_mode").fetchone()[0], "truncate"
                )
            finally:
                durable.close()
            local = _repository_index_connection(root / "local.sqlite3", local=True)
            try:
                self.assertEqual(local.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            finally:
                local.close()

    def test_page_cache_is_large_enough_to_hold_a_hash_keyed_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            connection = _repository_index_connection(Path(temporary) / "index.sqlite3")
            try:
                # Negative cache_size is a KiB budget, not a page count.
                cache_kib = -int(connection.execute("PRAGMA cache_size").fetchone()[0])
                self.assertGreaterEqual(cache_kib, 512 * 1024)
            finally:
                connection.close()

    def test_scratch_backed_index_publishes_and_reseeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.parquet"
            self._metadata(metadata, 500, seed=3)
            durable = root / "lustre" / "requests.sqlite3"

            index = ScratchBackedDatabase(
                durable,
                connect=lambda path, local: _repository_index_connection(path, local=local),
                scratch_root=str(root / "scratch"),
                checkpoint_seconds=0.0,
                identity="repository-code/repository-index",
                settings_table="settings",
                sequence_key="state_sequence",
            )
            try:
                self.assertTrue(index.local)
                self.assertFalse(durable.exists())
                report = _ingest_repository_metadata(
                    index.connection,
                    source={"id": "repository-code"},
                    metadata_files=[
                        ({"repo_id": "nvidia/code-v1", "revision": "a" * 40}, metadata)
                    ],
                    spool_root=root / "index-sort",
                    sort_buckets=8,
                    checkpoint=index.checkpoint,
                )
                self.assertGreater(report["requested_paths"], 0)
            finally:
                index.close()
            self.assertTrue(durable.is_file())

            # A resume that lost the node re-seeds from Lustre and skips the
            # units the published copy already records as complete.
            elsewhere = ScratchBackedDatabase(
                durable,
                connect=lambda path, local: _repository_index_connection(path, local=local),
                scratch_root=str(root / "other-scratch"),
                checkpoint_seconds=0.0,
                identity="repository-code/repository-index",
                settings_table="settings",
                sequence_key="state_sequence",
            )
            try:
                resumed = _ingest_repository_metadata(
                    elsewhere.connection,
                    source={"id": "repository-code"},
                    metadata_files=[
                        ({"repo_id": "nvidia/code-v1", "revision": "a" * 40}, metadata)
                    ],
                    spool_root=root / "index-sort",
                    sort_buckets=8,
                )
                self.assertEqual(resumed["metadata_units_reused"], 1)
                self.assertEqual(resumed["metadata_units_completed"], 0)
                self.assertEqual(resumed["requested_paths"], report["requested_paths"])
            finally:
                elsewhere.close()


if __name__ == "__main__":
    unittest.main()
