from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metis_data.local_download import _run_local_download_supervisor_locked
from metis_data.state import StateStore


class LocalDownloadSupervisorTests(unittest.TestCase):
    def test_transient_task_failure_retries_inside_same_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = StateStore(root / "state")
            task = {
                "task_id": "download-000000-test",
                "planned_bytes": 0,
                "items": [{"kind": "hf_file"}],
            }
            state.write("sources.lock.json", payload={"download_tasks": [task]})
            profile = {
                "manifest": "manifests/metis-1.6.yaml",
                "storage": {
                    "lustre_root": str(root),
                    "safety_free_tb": 0,
                    "directories": {"state": "state"},
                },
                "acquisition": {
                    "max_workers": 1,
                    "maximum_task_attempts": 3,
                    "retry_initial_seconds": 0,
                    "retry_maximum_seconds": 0,
                },
            }
            attempts = 0

            def transient_then_complete(_profile: dict, _task_index: int) -> dict:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise ConnectionError("temporary network failure")
                state.complete("download", task["task_id"], {"files": [{"kind": "hf_file"}]})
                return {"task_id": task["task_id"]}

            with (
                mock.patch(
                    "metis_data.local_download.run_download_task",
                    side_effect=transient_then_complete,
                ),
                mock.patch(
                    "metis_data.local_download.prepare_holdouts",
                    return_value={"status": "complete"},
                ),
                mock.patch(
                    "metis_data.local_download.write_acquisition_handoff",
                    return_value={"schema": "test-handoff"},
                ),
            ):
                result = _run_local_download_supervisor_locked(
                    root / "login2.yaml", profile, state
                )

            self.assertEqual(attempts, 2)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(result["retry_history"]), 1)
            self.assertEqual(result["retry_history"][0]["attempt"], 1)


if __name__ == "__main__":
    unittest.main()
