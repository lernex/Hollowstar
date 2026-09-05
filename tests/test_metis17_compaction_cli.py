from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metis_data17.cli import main


class Metis17CompactionControlTests(unittest.TestCase):
    def test_runtime_override_reaches_both_entry_points(self):
        with tempfile.TemporaryDirectory() as directory:
            for command, target in (
                ("prep", "metis_data17.worker.prep_service"),
                ("supervise-prep", "metis_data17.runtime.supervise_prep"),
            ):
                args = [command, "--root", directory, "--defer-compaction"]
                if command == "supervise-prep":
                    args.extend(["--python", sys.executable])
                with self.subTest(command=command), patch(target) as service:
                    self.assertEqual(main(args), 0)
                    self.assertIs(service.call_args.kwargs["defer_compaction"], True)

    def test_real_slurm_wrapper_rejects_misspelled_override(self):
        code = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(code / "slurm" / "metis17" / "prepare.sbatch")],
                env={
                    "PATH": os.environ["PATH"], "METIS17_ROOT": directory,
                    "METIS17_CODE": str(code), "METIS17_PYTHON": sys.executable,
                    "METIS17_WORKERS": "1", "METIS17_DEFER_COMPACTION": "tru",
                },
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid METIS17_DEFER_COMPACTION", result.stderr)


if __name__ == "__main__":
    unittest.main()
