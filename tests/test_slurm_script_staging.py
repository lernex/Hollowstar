from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metis_data.config import repository_root

SCRIPTS = ("stage.sbatch", "portage-stage.sbatch", "portage-family.sbatch")


def _stage_like_slurm(script: Path, destination: Path) -> Path:
    """Reproduce how Slurm delivers a batch script to a compute node.

    sbatch does not execute the file that was submitted. It copies the contents
    into /var/spool/slurmd/job<id>/slurm_script and runs that, so anything the
    script infers from its own path resolves under the spool directory instead
    of the checkout.
    """

    spool = destination / "var" / "spool" / "slurmd" / "job457528"
    spool.mkdir(parents=True)
    staged = spool / "slurm_script"
    shutil.copy2(script, staged)
    staged.chmod(0o755)
    return staged


class SlurmScriptStagingTests(unittest.TestCase):
    def test_stage_script_finds_the_checkout_when_run_from_the_spool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            staged = _stage_like_slurm(
                repository_root() / "slurm" / "metis16" / "stage.sbatch", workspace
            )
            # Stand in for the runtime interpreter so the script's own resolution
            # is what is under test, not the presence of a virtualenv.
            probe = workspace / "python-probe"
            probe.write_text(
                '#!/usr/bin/env bash\necho "PYTHONPATH=$PYTHONPATH"\necho "ARGV=$*"\n',
                encoding="utf-8",
            )
            probe.chmod(0o755)
            result = subprocess.run(
                ["bash", str(staged)],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "METIS_ROOT": str(repository_root()),
                    "METIS_PYTHON": str(probe),
                    "METIS_PROFILE": "/does/not/need/to/exist.yaml",
                    "METIS_STAGE": "handoff_signature",
                    "METIS_TASK_OFFSET": "0",
                    "METIS_TASKS_PER_JOB": "32",
                    "METIS_TASK_LIMIT": "1851",
                    "SLURM_ARRAY_TASK_ID": "1",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"PYTHONPATH={repository_root() / 'src'}", result.stdout)
            self.assertIn("--stage handoff_signature", result.stdout)
            # Array entry 1 owns global indices 32..63.
            self.assertIn("--task-index 32", result.stdout)
            self.assertNotIn("/var/spool", result.stdout)

    def test_stage_script_refuses_to_guess_the_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            staged = _stage_like_slurm(
                repository_root() / "slurm" / "metis16" / "stage.sbatch", workspace
            )
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("METIS_")
            }
            result = subprocess.run(
                ["bash", str(staged)],
                capture_output=True,
                text=True,
                env={
                    **environment,
                    "METIS_PROFILE": "/does/not/need/to/exist.yaml",
                    "METIS_STAGE": "handoff_signature",
                },
            )
            # Silently resolving to /var/spool is what cost a full submission.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("METIS_ROOT", result.stderr)

    def test_stage_script_names_a_missing_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            staged = _stage_like_slurm(
                repository_root() / "slurm" / "metis16" / "stage.sbatch", workspace
            )
            absent = workspace / "no-such-runtime" / "bin" / "python"
            result = subprocess.run(
                ["bash", str(staged)],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "METIS_ROOT": str(repository_root()),
                    "METIS_PYTHON": str(absent),
                    "METIS_PROFILE": "/does/not/need/to/exist.yaml",
                    "METIS_STAGE": "handoff_signature",
                },
            )
            self.assertNotEqual(result.returncode, 0)
            # Bash's bare "No such file or directory" cost a whole submission.
            self.assertIn(str(absent), result.stderr)
            self.assertIn("runtime interpreter", result.stderr)

    def test_no_batch_script_derives_the_checkout_from_its_own_path(self) -> None:
        for name in SCRIPTS:
            with self.subTest(script=name):
                text = (repository_root() / "slurm" / "metis16" / name).read_text(
                    encoding="utf-8"
                )
                code = "\n".join(
                    line for line in text.splitlines() if not line.lstrip().startswith("#")
                )
                self.assertNotIn("BASH_SOURCE", code)
                self.assertIn("METIS_ROOT", code)


class SlurmSubmissionExportsTests(unittest.TestCase):
    def test_every_submitted_stage_exports_the_checkout(self) -> None:
        from metis_data.slurm import submit_stage

        with tempfile.TemporaryDirectory() as temporary:
            profile = {
                "scheduler": {"partition": "parry", "exclusive_nodes": True},
                "storage": {
                    "lustre_root": temporary,
                    "directories": {"logs": "logs"},
                },
            }
            for stage in ("handoff_signature", "normalize", "tokenizer_train"):
                with self.subTest(stage=stage):
                    job = submit_stage(
                        stage=stage,
                        profile_path=Path("configs/metis16/portage-cpu.yaml"),
                        profile=profile,
                        dry_run=True,
                    )
                    export = next(
                        (arg for arg in job.command if arg.startswith("--export=")), None
                    )
                    self.assertIsNotNone(export)
                    self.assertIn(f"METIS_ROOT={repository_root()}", export)
                    # The runtime can live outside the checkout, so the stages
                    # must inherit the submitter's interpreter rather than
                    # re-deriving one from a default path.
                    self.assertIn(f"METIS_PYTHON={sys.executable}", export)
                    # One task the scheduler kills must not strand the other 51
                    # jobs behind an unsatisfiable afterok. Safe because every
                    # stage skips work that already has a completion marker.
                    self.assertIn("--requeue", job.command)

    def test_the_portage_launcher_exports_the_checkout(self) -> None:
        source = (repository_root() / "src" / "metis_portage" / "launcher.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"--export=ALL",', source)
        self.assertIn("METIS_ROOT=", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
