from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metis_data17.common import read_receipt, write_receipt
from metis_data17.runtime import idle_nodes, submit_prep_workers


class Metis17RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "metis-1.7-test"
        self.code = Path(self.tmp.name) / "code"
        script = self.code / "slurm" / "metis17" / "prepare.sbatch"
        script.parent.mkdir(parents=True)
        script.write_text("#!/usr/bin/env bash\nexit 0\n")

    def test_idle_node_enumeration_is_unique_and_partition_scoped(self):
        with patch("metis_data17.runtime.subprocess.check_output", return_value=(
            "node02|192|512000\nnode01|192|512000\nnode02|192|512000\n"
        )) as command:
            nodes = idle_nodes()
        self.assertEqual([node["name"] for node in nodes], ["node01", "node02"])
        self.assertIn("--partition=parry", command.call_args.args[0])

    def test_submit_only_idle_nodes_and_never_exports_parent_credentials(self):
        commands = []

        def output(argv, **_kwargs):
            commands.append(argv)
            if argv[0] == "git":
                return "a" * 40
            if argv[0] == "sbatch":
                return "101\n"
            raise AssertionError(argv)

        with patch("metis_data17.runtime.idle_nodes", return_value=[
            {"name": "node01", "cpus": 192, "memory_mb": 512000},
        ]), patch("metis_data17.runtime.subprocess.check_output", side_effect=output):
            result = submit_prep_workers(self.root, self.code, Path(sys.executable), maximum_nodes=4)
        self.assertEqual([job["job_id"] for job in result["active_jobs"]], ["101"])
        command = next(row for row in commands if row[0] == "sbatch")
        self.assertIn("--nodes=1", command)
        self.assertFalse(any(value.startswith("--nodelist=") for value in command))
        self.assertIn("--nice=10000", command)
        exports = next(value for value in command if value.startswith("--export="))
        self.assertNotIn("ALL", exports)
        self.assertNotIn("HF_TOKEN", exports)
        self.assertIn("METIS17_WORKERS=32", exports)
        registry = read_receipt(self.root / "state" / "prep-jobs.json")
        self.assertEqual(registry["jobs"][0]["code_commit"], "a" * 40)

    def test_active_jobs_are_not_duplicated_when_controller_restarts(self):
        write_receipt(self.root / "state" / "prep-jobs.json", {
            "jobs": [{"job_id": "123", "submitted_at": "2026-01-01T00:00:00+00:00"}],
        })

        def output(argv, **_kwargs):
            if argv[0] == "squeue":
                return "123\n"
            if argv[0] == "git":
                return "a" * 40
            raise AssertionError(f"Unexpected scheduler submission: {argv}")

        with patch("metis_data17.runtime.idle_nodes") as idle, patch(
            "metis_data17.runtime.subprocess.check_output", side_effect=output,
        ):
            result = submit_prep_workers(self.root, self.code, Path(sys.executable), maximum_nodes=1)
        idle.assert_not_called()
        self.assertEqual(len(result["active_jobs"]), 1)

    def test_no_idle_nodes_waits_without_interfering_with_running_work(self):
        with patch("metis_data17.runtime.idle_nodes", return_value=[]), patch(
            "metis_data17.runtime.subprocess.check_output",
        ) as command:
            result = submit_prep_workers(self.root, self.code, Path(sys.executable))
        command.assert_not_called()
        self.assertEqual(result["status"], "waiting_for_idle_nodes")


if __name__ == "__main__":
    unittest.main()
