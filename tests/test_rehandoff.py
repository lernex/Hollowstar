from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from metis_data import cli as cli_module
from metis_data.handoff import verify_acquisition_handoff, write_acquisition_handoff
from metis_data.runtime_lock import runtime_contract
from metis_data.source_lock import _repository_commit, source_lock_sha256
from metis_data.state import StateStore


def _build_acquisition(root: Path) -> tuple[StateStore, dict, dict, Path]:
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
    lock = {
        "schema": "metis.source-lock/v4",
        "release": "test-release",
        "sources": [{"id": "source", "candidate_tokens": 0}],
        "runtime_contract": runtime_contract(),
        "repository_commit": _repository_commit()[0],
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
    state.write("sources.lock.json", payload=lock)
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
    return state, profile, manifest, artifact


def _reresolve(state: StateStore) -> None:
    """Rewrite the lock the way a post-code-change re-resolve does.

    A code change invalidates the lock's repository binding, so the operator
    re-resolves and a new lock file lands with a new digest. Every task
    identity is preserved, exactly as re-resolving an unchanged manifest
    produces -- which is why the handoff it breaks is still true about the data.
    """

    lock = state.read("sources.lock.json")
    lock.pop("lock_sha256", None)
    lock["resolved_at"] = "2026-08-02T00:00:00Z"
    lock["lock_sha256"] = source_lock_sha256(lock)
    state.write("sources.lock.json", payload=lock)


class RehandoffTests(unittest.TestCase):
    def _run(self, profile: dict, manifest: dict, state: StateStore) -> None:
        with mock.patch.object(
            cli_module, "_context", return_value=(Path("profile.yaml"), profile, manifest, state)
        ), mock.patch.object(cli_module, "resolve_sources", return_value=state.read("sources.lock.json")):
            self.assertEqual(cli_module.cmd_rehandoff(Namespace(profile="login2")), 0)

    def test_reresolved_lock_breaks_the_handoff_and_rehandoff_rebinds_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _ = _build_acquisition(root)
            before = state.read("ACQUISITION_READY.json")

            _reresolve(state)
            with self.assertRaisesRegex(RuntimeError, "immutable source lock changed"):
                verify_acquisition_handoff(profile, manifest, state)

            self._run(profile, manifest, state)

            after = state.read("ACQUISITION_READY.json")
            # The provenance binding moved; nothing describing the data did.
            self.assertNotEqual(before["source_lock_sha256"], after["source_lock_sha256"])
            self.assertEqual(before["completion_markers_sha256"], after["completion_markers_sha256"])
            self.assertEqual(before["artifact_bytes"], after["artifact_bytes"])
            self.assertEqual(before["holdouts"]["sha256"], after["holdouts"]["sha256"])
            verify_acquisition_handoff(profile, manifest, state)

    def test_superseded_attestations_are_archived_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _ = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _reresolve(state)
            self._run(profile, manifest, state)

            archived = sorted(state.path("handoff-archive").glob("ACQUISITION_READY.*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), original)

    def test_rehandoff_refuses_when_an_acquisition_output_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, artifact = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _reresolve(state)
            # Re-attestation must not be a path for different bytes to enter a
            # build, however innocent the lock change that motivated it.
            artifact.write_text(
                '{"text":"licensed payload"}\n{"text":"smuggled"}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "output size changed"):
                self._run(profile, manifest, state)

            # The original attestation survives the refusal untouched.
            self.assertEqual(state.path("ACQUISITION_READY.json").read_bytes(), original)

    def test_rehandoff_refuses_when_the_holdouts_changed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _ = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _reresolve(state)
            # Holdouts are hashed fresh rather than compared against the
            # download ledger, so nothing upstream of the sealed-field check
            # would notice a rebuilt contamination bundle slipping in here.
            (root / "contamination" / "holdouts.jsonl").write_text(
                '{"text":"different holdout"}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "the acquired data changed"):
                self._run(profile, manifest, state)
            self.assertEqual(state.path("ACQUISITION_READY.json").read_bytes(), original)

    def test_rehandoff_restores_the_attestation_when_regeneration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _ = _build_acquisition(root)
            original = state.path("ACQUISITION_READY.json").read_bytes()

            _reresolve(state)
            # Holdouts missing mid-run is the realistic failure: the command
            # must not leave acquisition with no attestation at all.
            (root / "contamination" / "holdouts.jsonl").unlink()

            with self.assertRaises(RuntimeError):
                self._run(profile, manifest, state)
            self.assertEqual(state.path("ACQUISITION_READY.json").read_bytes(), original)

    def test_rehandoff_refuses_a_profile_without_the_acquisition_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "mount"
            root.mkdir()
            state, profile, manifest, _ = _build_acquisition(root)
            compute = {**profile, "name": "portage-cpu", "operator": {"roles": ["compute"]}}
            with self.assertRaisesRegex(RuntimeError, "cannot run the acquisition role"):
                self._run(compute, manifest, state)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
