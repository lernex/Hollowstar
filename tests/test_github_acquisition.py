from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from metis_data.acquisition.github import (
    _archive_file,
    _event_repository_identities,
    _iter_events,
    _repository_archive,
    materialize_github,
)
from metis_data.acquisition.io import iter_jsonl_zst
from metis_data.download import run_download_task
from metis_data.source_lock import _github_partition_items, resolve_sources
from metis_data.state import StateStore


def _github_source(driver: str = "github_repositories") -> dict:
    return {
        "id": f"fresh-{driver}",
        "category": "code",
        "phase_tokens": {"phase_a": 1, "phase_b": 0, "phase_c": 0},
        "access": {
            "type": "github_public",
            "cutoff_start": "2025-01-01",
            "cutoff_end": "2026-06-30",
        },
        "acquisition": {"driver": driver},
    }


class _FixtureDownloadClient:
    def __init__(
        self,
        gharchive: Path,
        repository: Path,
        *,
        repository_metadata: dict[str, dict] | None = None,
    ) -> None:
        self.gharchive = gharchive
        self.repository = repository
        self.repository_metadata = repository_metadata or {
            "example/repository": {
                "id": 1,
                "node_id": "R_fixture",
                "full_name": "example/repository",
                "fork": False,
                "mirror_url": None,
                "parent": None,
                "source": None,
                "updated_at": "2025-01-01T00:00:00Z",
            }
        }
        self.calls: list[str] = []

    def download(self, url: str, destination: Path, *, expected_size: int | None = None) -> Path:
        self.calls.append(url)
        source = self.gharchive if "data.gharchive.org" in url else self.repository
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise RuntimeError("fixture size mismatch")
        return destination

    def request(self, method: str, url: str, **_kwargs: object) -> mock.Mock:
        self.calls.append(url)
        prefix = "https://api.github.com/repos/"
        repo = url[len(prefix) :] if url.startswith(prefix) else ""
        response = mock.Mock()
        response.headers = {"Date": "Wed, 01 Jan 2025 00:00:00 GMT"}
        if repo not in self.repository_metadata:
            response.status_code = 404
            response.json.return_value = {"message": "Not Found"}
        else:
            response.status_code = 200
            response.json.return_value = self.repository_metadata[repo]
        response.raise_for_status.return_value = None
        return response


def _write_gharchive(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for event in events:
                compressed.write(
                    (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )


def _add_tar_text(bundle: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    bundle.addfile(member, io.BytesIO(payload))


def _write_repository_archive(path: Path, sha: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = f"repo-{sha}"
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
        _add_tar_text(
            bundle,
            f"{top}/LICENSE",
            'Permission is hereby granted, free of charge, to any person obtaining a copy. '
            'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.',
        )
        _add_tar_text(
            bundle,
            f"{top}/src/example.py",
            "def stable_example(value: int) -> int:\n"
            '    """Return a deterministic fixture value for repository acquisition tests."""\n'
            "    return value * 2\n\n"
            "assert stable_example(21) == 42\n",
        )


def _github_item(driver: str, *, source_id: str) -> dict:
    return {
        "kind": "builder",
        "source_id": source_id,
        "driver": driver,
        "access": {
            "type": "github_public",
            "cutoff_start": "2025-01-01",
            "cutoff_end": "2025-01-01",
        },
        "partition": {
            "id": "2025-01",
            "start": "2025-01-01",
            "end": "2025-01-01",
            "ordinal": 0,
            "total_partitions": 1,
            "days": 1,
        },
        "candidate_tokens": 1,
        "planned_download_bytes": 1,
    }


def _github_profile(root: Path) -> dict:
    return {
        "storage": {
            "lustre_root": str(root),
            "directories": {"cache": "cache"},
        },
        "runtime": {"download_retries": 0, "request_timeout_seconds": 1},
    }


def _materialization_signature(result: dict) -> dict:
    receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
    shard_hashes = [
        hashlib.sha256(Path(shard["path"]).read_bytes()).hexdigest()
        for shard in receipt["shards"]
    ]
    return {
        "records": receipt["records"],
        "text_characters": receipt["text_characters"],
        "estimated_tokens": receipt["estimated_tokens"],
        "partition": receipt["partition"],
        "candidate_token_target": receipt["candidate_token_target"],
        "counters": receipt["counters"],
        "incomplete_recovery": receipt["incomplete_recovery"],
        "gharchive_validator": receipt["gharchive_validator"],
        "codeload_validator": receipt["codeload_validator"],
        "shard_hashes": shard_hashes,
    }


class GitHubSourceLockTests(unittest.TestCase):
    def test_monthly_partitions_are_exact_and_targets_reconcile(self) -> None:
        source = _github_source()
        plan = {"candidate_tokens": 39_600_000_000, "planned_download_bytes": 29_700_000_000}
        items = _github_partition_items(source, plan)
        self.assertEqual(len(items), 18)
        self.assertEqual(items[0]["partition"]["id"], "2025-01")
        self.assertEqual(items[-1]["partition"]["id"], "2026-06")
        self.assertEqual(sum(item["candidate_tokens"] for item in items), plan["candidate_tokens"])
        self.assertEqual(sum(item["planned_download_bytes"] for item in items), plan["planned_download_bytes"])
        self.assertTrue(all(item["candidate_tokens"] > 0 for item in items))
        per_day = [item["candidate_tokens"] / item["partition"]["days"] for item in items]
        self.assertLess(max(per_day) - min(per_day), 1.0)

    def test_source_lock_contains_one_task_per_month_and_no_credentials(self) -> None:
        source = _github_source("github_discussions")
        source["access"].update(
            {
                "token": "github_pat_NEVER_WRITE_THIS",
                "authorization": "Bearer NEVER_WRITE_THIS",
                "headers": {"Authorization": "Bearer NEVER_WRITE_THIS"},
            }
        )
        plan = {
            "id": source["id"],
            "candidate_tokens": 15_400_000_000,
            "planned_download_bytes": 12_320_000_000,
        }
        manifest = {
            "_path": "/manifest/metis-1.6.yaml",
            "release": "test-release",
            "selection": {"seed": 1},
            "sources": [source],
        }
        profile = {"scheduler": {"download": {"target_bytes_per_task": 20_000_000_000}}}
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            with (
                mock.patch("metis_data.source_lock.candidate_plan", return_value={"sources": [plan]}),
                mock.patch("metis_data.source_lock.total_phase_tokens", return_value=7_000_000_000),
            ):
                lock = resolve_sources(manifest, profile, state)
        self.assertEqual(len(lock["download_tasks"]), 18)
        self.assertTrue(all(len(task["items"]) == 1 for task in lock["download_tasks"]))
        items = [task["items"][0] for task in lock["download_tasks"]]
        self.assertEqual([item["partition"]["id"] for item in items], [
            f"{year}-{month:02d}" for year, months in ((2025, range(1, 13)), (2026, range(1, 7))) for month in months
        ])
        serialized = json.dumps(lock, sort_keys=True)
        self.assertNotIn("NEVER_WRITE_THIS", serialized)
        self.assertNotIn("authorization", serialized.lower())
        self.assertEqual(sum(item["candidate_tokens"] for item in items), plan["candidate_tokens"])

    def test_download_dispatches_github_builder_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {
                "storage": {
                    "lustre_root": str(root),
                    "safety_free_tb": 0,
                    "directories": {"state": "state", "cache": "cache"},
                },
                "runtime": {"hf_home": "cache/huggingface"},
            }
            item = {
                "kind": "builder",
                "source_id": "fresh-github",
                "driver": "github_repositories",
                "access": {
                    "type": "github_public",
                    "cutoff_start": "2025-01-01",
                    "cutoff_end": "2026-06-30",
                },
                "partition": {"id": "2025-01", "start": "2025-01-01", "end": "2025-01-31"},
                "candidate_tokens": 100,
                "planned_download_bytes": 10,
            }
            state = StateStore(root / "state")
            state.write(
                "sources.lock.json",
                payload={
                    "download_tasks": [
                        {"task_id": "download-000000", "planned_bytes": 10, "items": [item]}
                    ]
                },
            )
            materialized = {
                "kind": "materialized_dataset",
                "source_id": "fresh-github",
                "local_path": str(root / "raw" / "fresh-github" / "2025-01"),
                "receipt": str(root / "raw" / "fresh-github" / "2025-01" / "ACQUISITION_RECEIPT.json"),
                "shards": [],
                "size": 0,
                "materialized": True,
                "ready_for_training_build": True,
            }
            with mock.patch(
                "metis_data.acquisition.github.materialize_github", return_value=materialized
            ) as materialize:
                result = run_download_task(profile, 0)
            materialize.assert_called_once_with(item, profile=profile, root=root)
            self.assertEqual(result["files"], [materialized])


class GitHubRepositoryIdentityTests(unittest.TestCase):
    def _run_repository_fixture(
        self,
        root: Path,
        *,
        events: list[dict],
        metadata: dict[str, dict],
        token: bool = True,
    ) -> tuple[dict, _FixtureDownloadClient]:
        hour = datetime(2025, 1, 1, tzinfo=timezone.utc)
        sha = "a" * 40
        gharchive = root / "fixtures" / "identity.json.gz"
        repository = root / "fixtures" / "identity.tar.gz"
        _write_gharchive(gharchive, events)
        _write_repository_archive(repository, sha)
        client = _FixtureDownloadClient(
            gharchive,
            repository,
            repository_metadata=metadata,
        )
        environment = {"GITHUB_TOKEN": "github-test-token"} if token else {}
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("metis_data.acquisition.github._public_client", return_value=client),
            mock.patch(
                "metis_data.acquisition.github._archive_hours",
                side_effect=lambda *_args: iter([hour]),
            ),
        ):
            result = materialize_github(
                _github_item("github_repositories", source_id="identity-test"),
                profile=_github_profile(root),
                root=root,
            )
        return result, client

    def test_minimal_gharchive_repo_object_is_not_negative_evidence(self) -> None:
        event = {
            "id": "push-1",
            "type": "PushEvent",
            "created_at": "2025-01-01T00:00:00Z",
            "repo": {"id": 1, "name": "example/repository"},
            "payload": {"head": "a" * 40},
        }
        self.assertEqual(list(_event_repository_identities(event)), [])

    def test_explicit_fork_event_excludes_repo_without_api_lookup(self) -> None:
        sha = "a" * 40
        events = [
            {
                "id": "fork-1",
                "type": "ForkEvent",
                "created_at": "2025-01-01T00:00:00Z",
                "repo": {"id": 1, "name": "origin/project"},
                "payload": {
                    "forkee": {
                        "id": 2,
                        "full_name": "aaa/fork",
                        "fork": True,
                        "mirror_url": None,
                        "parent": {"full_name": "origin/project"},
                    }
                },
            },
            {
                "id": "push-fork",
                "type": "PushEvent",
                "created_at": "2025-01-01T00:01:00Z",
                "repo": {"id": 2, "name": "aaa/fork"},
                "payload": {"head": sha},
            },
            {
                "id": "push-canonical",
                "type": "PushEvent",
                "created_at": "2025-01-01T00:02:00Z",
                "repo": {"id": 3, "name": "zzz/canonical"},
                "payload": {"head": sha},
            },
        ]
        metadata = {
            "zzz/canonical": {
                "id": 3,
                "node_id": "R_canonical",
                "full_name": "zzz/canonical",
                "fork": False,
                "mirror_url": None,
                "parent": None,
                "source": None,
                "updated_at": "2025-01-01T00:02:00Z",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            result, client = self._run_repository_fixture(
                Path(temporary),
                events=events,
                metadata=metadata,
            )
            receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["counters"]["repository_fork"], 1)
            self.assertFalse(
                any(url.endswith("/aaa/fork") for url in client.calls),
                client.calls,
            )
            rows = [
                row
                for shard in receipt["shards"]
                for row in iter_jsonl_zst(Path(shard["path"]))
            ]
            self.assertTrue(rows)
            self.assertEqual({row["repo"] for row in rows}, {"zzz/canonical"})
            self.assertTrue(all(row["repository_is_fork"] is False for row in rows))
            self.assertTrue(all(row["repository_is_mirror"] is False for row in rows))
            self.assertTrue(
                all(
                    row["repository_identity_source"].startswith("github-rest-v3:")
                    for row in rows
                )
            )
            self.assertTrue(
                all(
                    re.fullmatch(
                        r"[0-9a-f]{64}", row["repository_identity_sha256"]
                    )
                    for row in rows
                )
            )

    def test_api_identified_mirror_is_rejected_before_codeload(self) -> None:
        sha = "a" * 40
        events = [
            {
                "id": "push-mirror",
                "type": "PushEvent",
                "created_at": "2025-01-01T00:00:00Z",
                "repo": {"name": "aaa/mirror"},
                "payload": {"head": sha},
            },
            {
                "id": "push-canonical",
                "type": "PushEvent",
                "created_at": "2025-01-01T00:01:00Z",
                "repo": {"name": "zzz/canonical"},
                "payload": {"head": sha},
            },
        ]
        metadata = {
            "aaa/mirror": {
                "id": 1,
                "node_id": "R_mirror",
                "full_name": "aaa/mirror",
                "fork": False,
                "mirror_url": "https://upstream.example/project.git",
                "parent": None,
                "source": None,
            },
            "zzz/canonical": {
                "id": 2,
                "node_id": "R_canonical",
                "full_name": "zzz/canonical",
                "fork": False,
                "mirror_url": None,
                "parent": None,
                "source": None,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            result, client = self._run_repository_fixture(
                Path(temporary),
                events=events,
                metadata=metadata,
            )
            receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["counters"]["repository_mirror"], 1)
            mirror_api = [
                call
                for call in client.calls
                if call == "https://api.github.com/repos/aaa/mirror"
            ]
            self.assertEqual(len(mirror_api), 1)
            self.assertFalse(
                any(
                    "codeload.github.com/aaa/mirror" in call
                    for call in client.calls
                )
            )

    def test_unknown_identity_without_token_fails_closed(self) -> None:
        sha = "a" * 40
        events = [
            {
                "id": "push-1",
                "type": "PushEvent",
                "created_at": "2025-01-01T00:00:00Z",
                "repo": {"name": "example/repository"},
                "payload": {"head": sha},
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "GITHUB_TOKEN or GH_TOKEN"):
                self._run_repository_fixture(
                    Path(temporary),
                    events=events,
                    metadata={},
                    token=False,
                )


class GitHubCrashRecoveryTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path, datetime, str]:
        sha = "a" * 40
        hour = datetime(2025, 1, 1, tzinfo=timezone.utc)
        events = [
            {
                "id": "push-1",
                "type": "PushEvent",
                "created_at": "2025-01-01T00:00:00Z",
                "repo": {"name": "example/repository"},
                "payload": {"head": sha},
            },
            {
                "id": "issue-1",
                "type": "IssuesEvent",
                "created_at": "2025-01-01T00:05:00Z",
                "repo": {"name": "example/repository"},
                "payload": {
                    "issue": {
                        "title": "Deterministic crash recovery",
                        "body": (
                            "This substantive engineering discussion explains how a restartable data "
                            "pipeline preserves durable output while recovering from an interrupted "
                            "writer. It is deliberately longer than the minimum discussion threshold."
                        ),
                        "html_url": "https://github.com/example/repository/issues/1",
                    }
                },
            },
        ]
        gharchive = root / "fixtures" / "2025-01-01-0.json.gz"
        repository = root / "fixtures" / f"{sha}.tar.gz"
        _write_gharchive(gharchive, events)
        _write_repository_archive(repository, sha)
        return gharchive, repository, hour, sha

    def _materialize(
        self,
        root: Path,
        item: dict,
        gharchive: Path,
        repository: Path,
        hour: datetime,
    ) -> dict:
        client = _FixtureDownloadClient(gharchive, repository)
        with (
            mock.patch.dict(os.environ, {"GITHUB_TOKEN": "github-test-token"}),
            mock.patch("metis_data.acquisition.github._public_client", return_value=client),
            mock.patch(
                "metis_data.acquisition.github._archive_hours",
                side_effect=lambda *_args: iter([hour]),
            ),
        ):
            return materialize_github(item, profile=_github_profile(root), root=root)

    def test_corrupt_cached_gharchive_and_codeload_are_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gharchive, repository, hour, sha = self._fixtures(root)
            client = _FixtureDownloadClient(gharchive, repository)
            cache = root / "cache"

            cached_hour = _archive_file(client, cache / "gharchive", hour)
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(len(list(_iter_events(cached_hour))), 2)
            self.assertEqual(
                _archive_file(client, cache / "gharchive", hour),
                cached_hour,
            )
            self.assertEqual(len(client.calls), 1)
            cached_hour.write_bytes(b"not a gzip stream")

            repaired_hour = _archive_file(client, cache / "gharchive", hour)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(len(list(_iter_events(repaired_hour))), 2)

            cached_repository = _repository_archive(
                client,
                "example/repository",
                sha,
                cache / "repositories",
            )
            self.assertEqual(len(client.calls), 3)
            with tarfile.open(cached_repository, "r:gz") as bundle:
                self.assertGreater(len(bundle.getmembers()), 0)
            self.assertEqual(
                _repository_archive(
                    client,
                    "example/repository",
                    sha,
                    cache / "repositories",
                ),
                cached_repository,
            )
            self.assertEqual(len(client.calls), 3)
            cached_repository.write_bytes(b"not a tar stream")

            repaired_repository = _repository_archive(
                client,
                "example/repository",
                sha,
                cache / "repositories",
            )
            self.assertEqual(len(client.calls), 4)
            with tarfile.open(repaired_repository, "r:gz") as bundle:
                self.assertGreater(len(bundle.getmembers()), 0)

    def test_repository_month_rebuild_after_crash_matches_clean_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_root = Path(temporary)
            gharchive, repository, hour, _sha = self._fixtures(fixture_root)
            resumed_root = fixture_root / "resumed"
            clean_root = fixture_root / "clean"
            item = _github_item("github_repositories", source_id="fresh-repositories")
            client = _FixtureDownloadClient(gharchive, repository)

            with (
                mock.patch.dict(os.environ, {"GITHUB_TOKEN": "github-test-token"}),
                mock.patch("metis_data.acquisition.github._public_client", return_value=client),
                mock.patch(
                    "metis_data.acquisition.github._archive_hours",
                    side_effect=lambda *_args: iter([hour]),
                ),
                mock.patch(
                    "metis_data.acquisition.github.complete_materialization",
                    side_effect=RuntimeError("simulated crash before receipt"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    materialize_github(
                        item,
                        profile=_github_profile(resumed_root),
                        root=resumed_root,
                    )

            incomplete = resumed_root / "raw" / item["source_id"] / "2025-01"
            self.assertFalse((incomplete / "ACQUISITION_RECEIPT.json").exists())
            self.assertTrue((incomplete / "activity.sqlite3").exists())

            resumed = self._materialize(
                resumed_root, item, gharchive, repository, hour
            )
            clean = self._materialize(clean_root, item, gharchive, repository, hour)
            self.assertEqual(
                _materialization_signature(resumed),
                _materialization_signature(clean),
            )

    def test_discussion_seen_database_cannot_advance_past_durable_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture_root = Path(temporary)
            gharchive, repository, hour, _sha = self._fixtures(fixture_root)
            resumed_root = fixture_root / "resumed"
            clean_root = fixture_root / "clean"
            repository_item = _github_item(
                "github_repositories", source_id="fresh-repositories"
            )
            discussion_item = _github_item(
                "github_discussions", source_id="fresh-discussions"
            )

            # Populate each root's reviewed repository-license policy exactly as
            # the production wave ordering does before discussions begin.
            self._materialize(
                resumed_root, repository_item, gharchive, repository, hour
            )
            self._materialize(clean_root, repository_item, gharchive, repository, hour)

            client = _FixtureDownloadClient(gharchive, repository)
            with (
                mock.patch.dict(os.environ, {"GITHUB_TOKEN": "github-test-token"}),
                mock.patch("metis_data.acquisition.github._public_client", return_value=client),
                mock.patch(
                    "metis_data.acquisition.github._archive_hours",
                    side_effect=lambda *_args: iter([hour]),
                ),
                mock.patch(
                    "metis_data.acquisition.github.complete_materialization",
                    side_effect=RuntimeError("simulated crash after seen commit"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                    materialize_github(
                        discussion_item,
                        profile=_github_profile(resumed_root),
                        root=resumed_root,
                    )

            incomplete = (
                resumed_root / "raw" / discussion_item["source_id"] / "2025-01"
            )
            self.assertTrue((incomplete / "seen.sqlite3").exists())
            self.assertFalse((incomplete / "ACQUISITION_RECEIPT.json").exists())

            resumed = self._materialize(
                resumed_root, discussion_item, gharchive, repository, hour
            )
            clean = self._materialize(
                clean_root, discussion_item, gharchive, repository, hour
            )
            self.assertEqual(
                _materialization_signature(resumed),
                _materialization_signature(clean),
            )
            receipt = json.loads(Path(clean["receipt"]).read_text(encoding="utf-8"))
            discussion_rows = [
                row
                for shard in receipt["shards"]
                for row in iter_jsonl_zst(Path(shard["path"]))
            ]
            self.assertTrue(discussion_rows)
            self.assertTrue(
                all(row["repository_is_fork"] is False for row in discussion_rows)
            )
            self.assertTrue(
                all(row["repository_is_mirror"] is False for row in discussion_rows)
            )
            self.assertTrue(
                all(
                    re.fullmatch(
                        r"[0-9a-f]{64}", row["repository_identity_sha256"]
                    )
                    for row in discussion_rows
                )
            )


class GitHubHourlyArchiveBudgetTests(unittest.TestCase):
    """The GH Archive event walk must be boundable for both GitHub drivers.

    A monthly partition is about 720 hourly files and 70-140GiB, and neither
    driver can commit its month until the walk reaches the end of the window.
    The codeload cap bounds repository archives but not this, so an uncapped
    walk is still an open-ended transfer -- and for the discussion driver, whose
    gate is a repository slice that only the repository walk fills, an
    unbounded walk can move terabytes to accept nothing at all.
    """

    def _walk(self, driver: str, *, cap: int | None, days: int) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sha = "a" * 40
            gharchive = root / "fixtures" / "hours.json.gz"
            repository = root / "fixtures" / "hours.tar.gz"
            _write_gharchive(
                gharchive,
                [
                    {
                        "id": "push-1",
                        "type": "PushEvent",
                        "created_at": "2025-01-01T00:00:00Z",
                        "repo": {"id": 1, "name": "example/repository"},
                        "payload": {"head": sha},
                    }
                ],
            )
            _write_repository_archive(repository, sha)
            client = _FixtureDownloadClient(gharchive, repository)
            item = _github_item(driver, source_id=f"budget-{driver}")
            start = "2025-01-01"
            end = f"2025-01-{days:02d}"
            item["access"] = {**item["access"], "cutoff_start": start, "cutoff_end": end}
            item["partition"] = {**item["partition"], "start": start, "end": end}
            profile = _github_profile(root)
            if cap is not None:
                profile["acquisition"] = {"github_maximum_hourly_archives": cap}
            with (
                mock.patch.dict(
                    os.environ, {"GITHUB_TOKEN": "github-test-token"}, clear=True
                ),
                mock.patch(
                    "metis_data.acquisition.github._public_client", return_value=client
                ),
            ):
                result = materialize_github(item, profile=profile, root=root)
            receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
            return {
                "counters": receipt["counters"],
                "archive_urls": [
                    url for url in client.calls if "data.gharchive.org" in url
                ],
            }

    def test_an_uncapped_walk_reads_every_hour_in_the_partition(self) -> None:
        # Establishes what the budget is preventing: unbudgeted, the walk is
        # bounded only by the width of the window.
        walk = self._walk("github_repositories", cap=None, days=2)
        self.assertEqual(walk["counters"]["archives"], 48)
        self.assertEqual(len(walk["archive_urls"]), 48)
        self.assertNotIn("stopped_at_hourly_archive_budget", walk["counters"])

    def test_the_budget_stops_the_walk_and_is_recorded_in_the_receipt(self) -> None:
        walk = self._walk("github_repositories", cap=5, days=2)
        self.assertEqual(walk["counters"]["archives"], 5)
        # Bounded in transfers, not merely in accounting.
        self.assertEqual(len(walk["archive_urls"]), 5)
        # The receipt has to say the partition is short by policy, so a
        # truncated window is never read as an exhausted one.
        self.assertEqual(walk["counters"]["stopped_at_hourly_archive_budget"], 1)

    def test_a_budgeted_partition_that_accepts_nothing_still_fails_closed(
        self,
    ) -> None:
        """A budget bounds transfer; it does not license an empty partition.

        This is why withdrawing fresh repository code also had to withdraw
        engineering discussions.  The discussion driver accepts only events
        whose repository the repository walk has already licensed, so with that
        walk gone every discussion partition accepts nothing -- and an empty
        partition raises here rather than reporting a covered shortfall.  A
        budget cannot rescue that, and this test pins the reason so the source
        is not quietly restored on the assumption that it would degrade
        gracefully.
        """
        with self.assertRaisesRegex(RuntimeError, "produced no records"):
            self._walk("github_discussions", cap=5, days=2)


if __name__ == "__main__":
    unittest.main()
