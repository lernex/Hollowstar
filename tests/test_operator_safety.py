from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from metis_data import cli as cli_module
from metis_data.build_inputs import prepare_build_inputs
from metis_data.config import load_profile
from metis_data.doctor import _lustre_quota_check
from metis_data.local_download import _lane_configuration
from metis_data.download import run_download_task
from metis_data.handoff import (
    _validate_materialized_token_targets,
    verify_acquisition_handoff,
    write_acquisition_handoff,
)
from metis_data.runtime_lock import runtime_contract
from metis_data.source_lock import _select_files, resolve_sources, source_lock_sha256
from metis_data.state import StateStore, atomic_json


class OperatorSafetyTests(unittest.TestCase):
    def test_take_all_and_handoff_preserve_every_source_shortfall(self) -> None:
        selected, short = _select_files(
            [{"path": "only.parquet", "size": 10}],
            seed=1,
            repo_id="example/source",
            target_bytes=20,
            take_all=True,
        )
        self.assertEqual(len(selected), 1)
        self.assertTrue(short)

        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            lock = {
                "sources": [
                    {"id": "web-source", "driver": "hf_snapshot", "candidate_tokens": 100},
                    {"id": "code-source", "driver": "repository_index", "candidate_tokens": 50},
                ],
                "download_tasks": [
                    {"task_id": "download-000000", "items": []},
                ],
            }
            state.complete(
                "download",
                "download-000000",
                {
                    "files": [
                        {
                            "source_id": "web-source",
                            "candidate_token_estimate": 100,
                            "candidate_estimator": "test",
                        },
                        {
                            "source_id": "code-source",
                            "payload_role": "source_index",
                            "candidate_token_estimate": 1_000,
                        },
                        {
                            "source_id": "code-source",
                            "candidate_token_estimate": 49,
                            "candidate_estimator": "accepted_code",
                        },
                    ]
                },
            )
            manifest = {
                "sources": [
                    {"id": "web-source", "category": "web"},
                    {"id": "code-source", "category": "code"},
                ]
            }
            with self.assertRaisesRegex(RuntimeError, "code-source=49/50"):
                _validate_materialized_token_targets(lock, state, manifest)

    def test_source_lock_rejects_known_take_all_shortfall_before_download(self) -> None:
        source = {
            "id": "bounded-source",
            "category": "reference",
            "phase_tokens": {"phase_a": 1, "phase_b": 0, "phase_c": 0},
            "access": {
                "type": "huggingface",
                "repo_id": "example/bounded",
                "revision": "a" * 40,
                "allow_patterns": ["**/*.parquet"],
                "take_all": True,
            },
            "acquisition": {
                "driver": "hf_snapshot",
                "compressed_bytes_per_token": 1.0,
            },
        }
        manifest = {
            "_path": "/manifest/metis-1.6.yaml",
            "release": "test-release",
            "selection": {"seed": 1},
            "sources": [source],
        }
        plan = {
            "id": source["id"],
            "candidate_tokens": 20,
            "planned_download_bytes": 20,
        }
        api = SimpleNamespace(
            dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40),
            list_repo_tree=lambda *args, **kwargs: [
                SimpleNamespace(path="data/only.parquet", size=10, blob_id="blob", lfs=None)
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            with (
                mock.patch("metis_data.source_lock.HfApi", return_value=api),
                mock.patch("metis_data.source_lock.get_token", return_value=None),
                mock.patch("metis_data.source_lock.candidate_plan", return_value={"sources": [plan]}),
                mock.patch("metis_data.source_lock.total_phase_tokens", return_value=1),
            ):
                with self.assertRaisesRegex(RuntimeError, "take_all=True.*not permit"):
                    resolve_sources(manifest, {"scheduler": {"download": {}}}, state)

    def test_source_lock_is_manifest_bound_and_tasks_are_content_addressed(self) -> None:
        source = {
            "id": "fresh-code",
            "category": "code",
            "phase_tokens": {"phase_a": 1, "phase_b": 0, "phase_c": 0},
            "access": {
                "type": "github_public",
                "cutoff_start": "2026-01-01",
                "cutoff_end": "2026-01-31",
            },
            "acquisition": {"driver": "github_repositories"},
        }
        manifest = {
            "_path": "/manifest/metis-1.6.yaml",
            "release": "test-release",
            "selection": {"seed": 1},
            "sources": [source],
        }
        plan = {
            "id": "fresh-code",
            "candidate_tokens": 1000,
            "planned_download_bytes": 2000,
        }
        profile = {"scheduler": {"download": {"target_bytes_per_task": 10_000}}}
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            with (
                mock.patch("metis_data.source_lock.candidate_plan", return_value={"sources": [plan]}),
                mock.patch("metis_data.source_lock.total_phase_tokens", return_value=1),
            ):
                lock = resolve_sources(manifest, profile, state)
                self.assertEqual(lock["schema"], "metis.source-lock/v4")
                self.assertEqual(lock["runtime_contract"], runtime_contract())
                self.assertEqual(lock["lock_sha256"], source_lock_sha256(lock))
                task = lock["download_tasks"][0]
                self.assertRegex(task["task_id"], r"^download-000000-[0-9a-f]{16}$")
                self.assertEqual(len(task["task_sha256"]), 64)
                changed = {**manifest, "selection": {"seed": 2}}
                with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                    resolve_sources(changed, profile, state)

                tampered = json.loads(json.dumps(lock))
                tampered["download_tasks"][0]["items"][0]["candidate_tokens"] += 1
                state.write("sources.lock.json", payload=tampered)
                with self.assertRaisesRegex(RuntimeError, "whole-lock self-hash"):
                    resolve_sources(manifest, profile, state)

                bad_identity = json.loads(json.dumps(lock))
                bad_identity["download_tasks"][0]["task_id"] = "download-000000-badbadbadbadbadb"
                bad_identity["lock_sha256"] = source_lock_sha256(bad_identity)
                state.write("sources.lock.json", payload=bad_identity)
                with self.assertRaisesRegex(RuntimeError, "task identity mismatch"):
                    resolve_sources(manifest, profile, state)

    def test_run_acquisition_propagates_failure_and_requires_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            context = (
                Path("/profile.yaml"),
                {"name": "login2"},
                {"release": "test-release"},
                state,
            )
            args = Namespace(profile="login2")
            with (
                mock.patch.object(cli_module, "_context", return_value=context),
                mock.patch.object(cli_module, "_require_acquisition_preflight"),
                mock.patch.object(cli_module, "resolve_sources"),
                mock.patch.object(cli_module, "_print"),
                mock.patch.object(
                    cli_module,
                    "run_local_download_supervisor",
                    return_value={"status": "failed", "task_failures": [{"task_index": 1}]},
                ),
            ):
                self.assertEqual(cli_module.cmd_run_acquisition(args), 1)

            with (
                mock.patch.object(cli_module, "_context", return_value=context),
                mock.patch.object(cli_module, "_require_acquisition_preflight"),
                mock.patch.object(cli_module, "resolve_sources"),
                mock.patch.object(cli_module, "_print"),
                mock.patch.object(
                    cli_module,
                    "run_local_download_supervisor",
                    return_value={"status": "complete"},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "without ACQUISITION_READY.json"):
                    cli_module.cmd_run_acquisition(args)

            state.write("ACQUISITION_READY.json", payload={"release": "test-release"})
            with (
                mock.patch.object(cli_module, "_context", return_value=context),
                mock.patch.object(cli_module, "_require_acquisition_preflight"),
                mock.patch.object(cli_module, "resolve_sources"),
                mock.patch.object(cli_module, "_print"),
                mock.patch.object(
                    cli_module,
                    "run_local_download_supervisor",
                    return_value={
                        "status": "complete",
                        "acquisition_handoff": {"release": "test-release"},
                    },
                ),
            ):
                self.assertEqual(cli_module.cmd_run_acquisition(args), 0)

    def test_ambiguous_production_quota_requires_explicit_acknowledgement(self) -> None:
        quota_output = (
            "Disk quotas for usr account (uid 1):\n"
            "     Filesystem  kbytes quota limit grace files quota limit grace\n"
            "/lus/lustre1/vollmerc 0 0 0 - 0 0 0 -\n"
        )
        command_result = SimpleNamespace(stdout=quota_output)
        root = Path("/lus/lustre1/vollmerc/metis-1.6")
        profile = {
            "storage": {
                "require_explicit_quota_acknowledgement": True,
                "minimum_free_tb": 25,
                "minimum_free_inodes": 100_000,
            }
        }
        with (
            mock.patch("metis_data.doctor.shutil.which", return_value="/usr/bin/lfs"),
            mock.patch("metis_data.doctor.subprocess.run", return_value=command_result),
        ):
            missing = _lustre_quota_check(profile, root)
            self.assertEqual(missing.status, "FAIL")
            self.assertIn("--quota-acknowledgement", missing.detail)

            confirmed = {
                "storage": {
                    **profile["storage"],
                    "quota_unknown_acknowledgement": "administrator-confirmed",
                }
            }
            accepted = _lustre_quota_check(confirmed, root)
            self.assertEqual(accepted.status, "PASS")
            self.assertIn("administrator-confirmed", accepted.detail)

            invalid = {
                "storage": {
                    **profile["storage"],
                    "quota_unknown_acknowledgement": "yes",
                }
            }
            rejected = _lustre_quota_check(invalid, root)
            self.assertEqual(rejected.status, "FAIL")
            self.assertIn("unsupported", rejected.detail)

        with mock.patch.dict(
            os.environ,
            {
                "METIS_LUSTRE_ROOT": str(root),
                "METIS_LUSTRE_QUOTA_ACKNOWLEDGEMENT": "unlimited",
            },
        ):
            _, loaded = load_profile("login2")
        self.assertEqual(
            loaded["storage"]["quota_unknown_acknowledgement"],
            "unlimited",
        )

    def test_huggingface_token_file_must_be_private_and_not_a_symlink(self) -> None:
        validator = Path(__file__).resolve().parents[1] / "ops" / "validate-hf-token-file.sh"
        with tempfile.TemporaryDirectory() as temporary:
            token = Path(temporary) / "token"
            token.write_text("hf_test_token\n", encoding="utf-8")
            token.chmod(0o600)
            accepted = subprocess.run(
                [str(validator), str(token)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

            token.chmod(0o640)
            exposed = subprocess.run(
                [str(validator), str(token)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(exposed.returncode, 0)
            self.assertIn("permissions", exposed.stderr)

            token.chmod(0o700)
            executable = subprocess.run(
                [str(validator), str(token)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(executable.returncode, 0)
            self.assertIn("permissions", executable.stderr)

            token.chmod(0o600)
            link = Path(temporary) / "token-link"
            link.symlink_to(token)
            symlinked = subprocess.run(
                [str(validator), str(link)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(symlinked.returncode, 0)
            self.assertIn("symbolic link", symlinked.stderr)

    def test_download_completion_must_match_task_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = StateStore(root / "state")
            task_id = "download-000000-deadbeefdeadbeef"
            state.write(
                "sources.lock.json",
                payload={
                    "download_tasks": [
                        {"task_id": task_id, "task_sha256": "a" * 64, "items": []}
                    ]
                },
            )
            state.complete("download", task_id, {"task_sha256": "b" * 64, "files": []})
            profile = {
                "storage": {
                    "lustre_root": str(root),
                    "directories": {"state": "state"},
                }
            }
            with self.assertRaisesRegex(RuntimeError, "immutable task"):
                run_download_task(profile, 0)

    def test_download_resume_reconciles_completed_artifacts_before_skipping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = StateStore(root / "state")
            artifact = root / "raw" / "source" / "part.jsonl"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"text":"candidate"}\n', encoding="utf-8")
            task_id = "download-000000-deadbeefdeadbeef"
            state.write(
                "sources.lock.json",
                payload={
                    "download_tasks": [
                        {"task_id": task_id, "task_sha256": "a" * 64, "items": []}
                    ]
                },
            )
            state.complete(
                "download",
                task_id,
                {
                    "task_sha256": "a" * 64,
                    "files": [
                        {
                            "kind": "materialized_jsonl",
                            "source_id": "source",
                            "local_path": str(artifact),
                            "size": artifact.stat().st_size,
                        }
                    ],
                },
            )
            profile = {
                "storage": {
                    "lustre_root": str(root),
                    "directories": {"state": "state"},
                }
            }
            artifact.unlink()
            with self.assertRaisesRegex(RuntimeError, "output is missing"):
                run_download_task(profile, 0)

    def test_handoff_and_build_inputs_rebase_to_a_different_mount_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            login_root = workspace / "login-mount"
            login_root.mkdir()
            artifact = login_root / "raw" / "source" / "payload.jsonl"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"text":"licensed payload"}\n', encoding="utf-8")
            contamination = login_root / "contamination"
            contamination.mkdir()
            (contamination / "holdouts.jsonl").write_text('{"text":"holdout"}\n', encoding="utf-8")
            (contamination / "HOLDOUTS.json").write_text(
                '{"schema":"metis.holdout-bundle/test"}\n', encoding="utf-8"
            )
            state = StateStore(login_root / "state")
            task_payload = {"items": [], "planned_bytes": 0}
            task_sha256 = hashlib.sha256(
                json.dumps(
                    task_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            task_id = f"download-000000-{task_sha256[:16]}"
            lock = {
                "schema": "metis.source-lock/v4",
                "release": "test-release",
                "sources": [{"id": "source", "candidate_tokens": 0}],
                "runtime_contract": runtime_contract(),
                "download_tasks": [
                    {
                        **task_payload,
                        "task_index": 0,
                        "task_sha256": task_sha256,
                        "task_id": task_id,
                    }
                ],
            }
            lock["lock_sha256"] = source_lock_sha256(lock)
            state.write(
                "sources.lock.json",
                payload=lock,
            )
            state.complete(
                "download",
                task_id,
                {
                    "task_sha256": task_sha256,
                    "files": [
                        {
                            "kind": "materialized_jsonl",
                            "source_id": "source",
                            "local_path": str(artifact),
                            "size": artifact.stat().st_size,
                        }
                    ]
                },
            )
            directories = {"state": "state", "contamination": "contamination"}
            login_profile = {
                "storage": {"lustre_root": str(login_root), "directories": directories},
                "gates": {"require_clean_repository": False},
            }
            manifest = {
                "release": "test-release",
                "sources": [{"id": "source", "category": "web"}],
            }
            write_acquisition_handoff(login_profile, manifest, state)

            rhea_root = workspace / "rhea-mount"
            shutil.copytree(login_root, rhea_root, copy_function=shutil.copy2)
            shutil.rmtree(login_root)
            rhea_state = StateStore(rhea_root / "state")
            rhea_profile = {
                "storage": {"lustre_root": str(rhea_root), "directories": directories},
                "gates": {
                    "require_repository_commit_match": False,
                    "allow_relocated_lustre_root": True,
                },
            }
            verified = verify_acquisition_handoff(
                rhea_profile, manifest, rhea_state, verify_artifact_hashes=True
            )
            self.assertTrue(verified["relocated_mount"])
            inputs = prepare_build_inputs(rhea_profile, rhea_state)
            self.assertEqual(inputs["input_count"], 1)
            self.assertTrue(inputs["inputs"][0]["local_path"].startswith(str(rhea_root.resolve())))

    def test_stale_lock_cleanup_preserves_a_live_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = StateStore(Path(temporary) / "state")
            lock = state.path("locks", "download", "task.lock")
            lock.mkdir(parents=True)
            atomic_json(
                lock / "OWNER.json",
                {"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": "old"},
            )
            old = time.time() - 10_000
            os.utime(lock, (old, old))
            self.assertEqual(state.clear_stale_locks(60), [])
            self.assertTrue(lock.is_dir())


class AcquisitionLaneTests(unittest.TestCase):
    def test_every_configured_lane_is_gated(self) -> None:
        """A lane with no configured limit runs ungated, not serialized.

        ``_run_task_in_lanes`` acquires ``semaphores.get(lane)`` and skips a
        lane it cannot find, so routing a driver to a lane that
        ``lane_max_workers`` never names silently removes its concurrency
        bound instead of failing.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "metis-1.6"
            root.mkdir(parents=True)
            with mock.patch.dict(
                os.environ,
                {
                    "METIS_LUSTRE_ROOT": str(root),
                    "METIS_LUSTRE_QUOTA_ACKNOWLEDGEMENT": "unlimited",
                },
            ):
                with mock.patch(
                    "metis_data.config.validate_storage_root", side_effect=lambda _profile, path: path
                ):
                    _, profile = load_profile("login2")

        driver_lanes, semaphores = _lane_configuration(profile)
        ungated = sorted(set(driver_lanes.values()) - set(semaphores))
        self.assertEqual(ungated, [], f"lanes without a configured limit: {ungated}")
        for lane in ("github", "github_discussions", "common_crawl"):
            self.assertIn(lane, semaphores)
        # Discussions read the REST API while repositories pull ~1GiB codeload
        # archives. Sharing a lane starved the discussion builder completely,
        # so keep them provably separate rather than merely both configured.
        self.assertNotEqual(
            driver_lanes["github_discussions"], driver_lanes["github_repositories"]
        )


if __name__ == "__main__":
    unittest.main()
