from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest import mock

from metis_data import cli as cli_module
from metis_data.handoff import verify_acquisition_handoff, write_acquisition_handoff
from metis_data.runtime_lock import runtime_contract
from metis_data.source_lock import _repository_commit, source_lock_sha256
from metis_data.state import StateStore

FOREIGN_COMMIT = "b" * 40


def _lock(commit: str, task_payload: dict, task_sha256: str, task_id: str) -> dict:
    lock = {
        "schema": "metis.source-lock/v4",
        "release": "test-release",
        "sources": [{"id": "source", "candidate_tokens": 0}],
        "runtime_contract": runtime_contract(),
        "repository_commit": commit,
        "resolved_at": "2026-07-25T00:00:00Z",
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
    return lock


def _build_acquisition(root: Path) -> tuple[StateStore, dict, dict, Path, dict]:
    """A minimal completed acquisition: one artifact, holdouts, a valid lock."""

    artifact = root / "raw" / "source" / "payload.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"text":"licensed payload"}\n', encoding="utf-8")
    contamination = root / "contamination"
    contamination.mkdir()
    (contamination / "holdouts.jsonl").write_text('{"text":"holdout"}\n', encoding="utf-8")
    (contamination / "HOLDOUTS.json").write_text(
        '{"schema":"metis.holdout-bundle/test"}\n', encoding="utf-8"
    )
    state = StateStore(root / "state")
    task_payload = {"items": [], "planned_bytes": 0}
    task_sha256 = hashlib.sha256(
        json.dumps(
            task_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    task_id = f"download-000000-{task_sha256[:16]}"
    identity = {
        "task_payload": task_payload,
        "task_sha256": task_sha256,
        "task_id": task_id,
    }
    state.write("sources.lock.json", payload=_lock(_repository_commit()[0], **identity))
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
            ],
        },
    )
    directories = {"state": "state", "contamination": "contamination"}
    profile = {
        "name": "login2",
        "operator": {"roles": ["acquisition"]},
        "storage": {"lustre_root": str(root), "directories": directories},
        "gates": {"require_clean_repository": False},
    }
    manifest = {"release": "test-release", "sources": [{"id": "source", "category": "web"}]}
    write_acquisition_handoff(profile, manifest, state)
    return state, profile, manifest, artifact, identity


def _rewrite_lock(state: StateStore, identity: dict, *, commit: str, resolved_at: str) -> None:
    lock = _lock(commit, **identity)
    lock.pop("lock_sha256")
    lock["resolved_at"] = resolved_at
    lock["lock_sha256"] = source_lock_sha256(lock)
    state.write("sources.lock.json", payload=lock)


class RehandoffTests(unittest.TestCase):
    def _run(self, profile: dict, manifest: dict, state: StateStore, identity: dict) -> None:
        """Drive cmd_rehandoff against a resolver with the real contract.

        `resolve_sources` never rebuilds a lock that already exists -- it only
        validates one, and refuses outright when the repository commit moved.
        Mocking that away hides the failure this command exists to survive, so
        the stand-in reproduces it and reaches the Hub for nothing else.
        """

        calls: list[str] = []

        def resolver(_manifest: dict, _profile: dict, store: StateStore) -> Any:
            existing = store.read("sources.lock.json")
            if existing is not None:
                calls.append("validated")
                if existing.get("repository_commit") != _repository_commit()[0]:
                    raise RuntimeError(
                        "The repository commit changed after the immutable source "
                        "lock was created"
                    )
                return existing
            calls.append("resolved")
            fresh = _lock(_repository_commit()[0], **identity)
            fresh.pop("lock_sha256")
            fresh["resolved_at"] = "2026-08-02T12:00:00Z"
            fresh["lock_sha256"] = source_lock_sha256(fresh)
            store.write("sources.lock.json", payload=fresh)
            return fresh

        with mock.patch.object(
            cli_module, "_context", return_value=(Path("profile.yaml"), profile, manifest, state)
        ), mock.patch.object(cli_module, "resolve_sources", side_effect=resolver):
            self.assertEqual(cli_module.cmd_rehandoff(Namespace(profile="login2")), 0)
        self._resolver_calls = calls

    def _archived_locks(self, state: StateStore) -> list[Path]:
        return sorted(state.path("handoff-archive").glob("sources.lock.*.json"))

    def test_a_lock_stale_against_this_checkout_is_archived_and_re_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            before = state.read("ACQUISITION_READY.json")

            # A code change lands: the lock is now bound to a commit that is no
            # longer HEAD, which is what `git pull` before `submit build` does.
            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            stale_lock = state.path("sources.lock.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "immutable source lock changed"):
                verify_acquisition_handoff(profile, manifest, state)

            self._run(profile, manifest, state, identity)

            self.assertEqual(self._resolver_calls, ["resolved"])
            archived = self._archived_locks(state)
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), stale_lock)
            after = state.read("ACQUISITION_READY.json")
            self.assertNotEqual(before["source_lock_sha256"], after["source_lock_sha256"])
            self.assertEqual(before["completion_markers_sha256"], after["completion_markers_sha256"])
            verify_acquisition_handoff(profile, manifest, state)

    def test_a_lock_valid_for_this_checkout_is_reused_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            before = state.read("ACQUISITION_READY.json")

            # Already re-resolved against this checkout, so only the handoff is
            # stale. Re-resolving again would be Hub work for nothing.
            _rewrite_lock(
                state,
                identity,
                commit=_repository_commit()[0],
                resolved_at="2026-08-02T00:00:00Z",
            )
            current_lock = state.path("sources.lock.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "immutable source lock changed"):
                verify_acquisition_handoff(profile, manifest, state)

            self._run(profile, manifest, state, identity)

            self.assertEqual(self._resolver_calls, ["validated"])
            self.assertEqual(self._archived_locks(state), [])
            self.assertEqual(state.path("sources.lock.json").read_bytes(), current_lock)
            after = state.read("ACQUISITION_READY.json")
            self.assertNotEqual(before["source_lock_sha256"], after["source_lock_sha256"])
            verify_acquisition_handoff(profile, manifest, state)

    def test_markers_signed_against_the_old_handoff_are_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            # Stand in for a completed deep-verification pass. Both of these
            # carry handoff_sha256 and raise rather than recompute when it
            # moves, so a re-attestation that leaves them behind produces a
            # submission that fails two commands later.
            state.write("completed", "handoff_signature", "task-000000-abc.json", payload={"n": 1})
            state.write("completed", "handoff_verify", "task-000000.json", payload={"n": 1})
            state.write("HANDOFF_VERIFIED.json", payload={"n": 1})

            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            self._run(profile, manifest, state, identity)

            self.assertFalse(state.path("completed", "handoff_signature").exists())
            self.assertFalse(state.path("completed", "handoff_verify").exists())
            self.assertFalse(state.path("HANDOFF_VERIFIED.json").exists())
            # Cleared means archived, never deleted.
            archived = sorted(p.name for p in state.path("handoff-archive").iterdir())
            self.assertTrue(any(name.startswith("handoff_signature.") for name in archived))
            self.assertTrue(any(name.startswith("handoff_verify.") for name in archived))
            self.assertTrue(any(name.startswith("HANDOFF_VERIFIED.") for name in archived))

    def test_superseded_attestations_are_archived_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            self._run(profile, manifest, state, identity)

            archived = sorted(
                state.path("handoff-archive").glob("ACQUISITION_READY.*.json")
            )
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), original)

    def test_rehandoff_refuses_when_an_acquisition_output_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, artifact, identity = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            stale_lock = state.path("sources.lock.json").read_bytes()
            # Re-attestation must not be a path for different bytes to enter a
            # build, however innocent the lock change that motivated it.
            artifact.write_text(
                '{"text":"licensed payload"}\n{"text":"smuggled"}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "output size changed"):
                self._run(profile, manifest, state, identity)

            # Both artifacts survive the refusal exactly as they were.
            self.assertEqual(state.path("ACQUISITION_READY.json").read_bytes(), original)
            self.assertEqual(state.path("sources.lock.json").read_bytes(), stale_lock)

    def test_a_manifest_metadata_correction_is_carried_not_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            before = state.read("ACQUISITION_READY.json")

            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            # A licence status decides how a record is treated, never which
            # bytes were fetched. Re-attestation has to carry that, or the only
            # way to correct a policy error is to redo an eleven-day
            # acquisition.
            manifest["sources"][0]["license"] = {
                "status": "reviewed",
                "expression": "ODC-By-1.0",
            }

            self._run(profile, manifest, state, identity)

            after = state.read("ACQUISITION_READY.json")
            self.assertNotEqual(before["manifest_sha256"], after["manifest_sha256"])
            # Everything describing the acquired bytes is still identical.
            self.assertEqual(before["completion_markers_sha256"], after["completion_markers_sha256"])
            self.assertEqual(before["artifact_bytes"], after["artifact_bytes"])
            self.assertEqual(before["holdouts"]["sha256"], after["holdouts"]["sha256"])
            verify_acquisition_handoff(profile, manifest, state)

    def test_a_manifest_change_that_moves_a_download_task_is_still_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            # Not sealing manifest_sha256 must not mean a manifest edit can
            # quietly rebind the attestation to work that was never done. A
            # re-resolve that moves a task identity has no completion marker.
            moved = dict(identity)
            moved["task_sha256"] = "f" * 64
            moved["task_id"] = "download-000000-" + "f" * 16

            with self.assertRaisesRegex(RuntimeError, "task is incomplete"):
                self._run(profile, manifest, state, moved)
            self.assertEqual(state.path("ACQUISITION_READY.json").read_bytes(), original)

    def test_rehandoff_refuses_when_the_holdouts_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            # Holdouts are hashed fresh rather than checked against the download
            # ledger, so nothing upstream of the sealed-field comparison would
            # notice a rebuilt contamination bundle slipping in here.
            (root / "contamination" / "holdouts.jsonl").write_text(
                '{"text":"different holdout"}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "the acquired data changed"):
                self._run(profile, manifest, state, identity)
            self.assertEqual(state.path("ACQUISITION_READY.json").read_bytes(), original)

    def test_rehandoff_restores_both_artifacts_when_regeneration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _rewrite_lock(
                state, identity, commit=FOREIGN_COMMIT, resolved_at="2026-07-25T00:00:00Z"
            )
            stale_lock = state.path("sources.lock.json").read_bytes()
            # Holdouts missing mid-run is the realistic failure: the command must
            # not leave acquisition with a re-resolved lock and no attestation.
            (root / "contamination" / "holdouts.jsonl").unlink()

            with self.assertRaises(RuntimeError):
                self._run(profile, manifest, state, identity)
            self.assertEqual(state.path("ACQUISITION_READY.json").read_bytes(), original)
            self.assertEqual(state.path("sources.lock.json").read_bytes(), stale_lock)

    def test_rehandoff_refuses_a_profile_without_the_acquisition_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _, identity = _build_acquisition(root)
            compute = {**profile, "name": "portage-cpu", "operator": {"roles": ["compute"]}}
            with self.assertRaisesRegex(RuntimeError, "cannot run the acquisition role"):
                self._run(compute, manifest, state, identity)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
