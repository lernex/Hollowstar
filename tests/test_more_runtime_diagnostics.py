from __future__ import annotations

import ast
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/metis_ablation/runtime_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("more_runtime_diagnostics_test_module", MODULE_PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class RuntimeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".runtime-diagnostics-tests-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.proc = self.root / "proc"
        boot = self.proc / "sys/kernel/random"
        boot.mkdir(parents=True)
        self.boot_id = str(uuid.uuid4())
        (boot / "boot_id").write_text(self.boot_id + "\n")
        (self.proc / "stat").write_text("cpu 1 2 3 4\nbtime 1700000000\nprocesses 42\n")
        self.run = self.root / "run"

    def tearDown(self):
        shutil.rmtree(self.root)

    def startup(self, rank=0, world_size=2, job_id="495851"):
        return diagnostics.record_rank_startup(
            self.run, rank=rank, world_size=world_size, local_rank=rank,
            device=f"cuda:{rank}", proc_root=self.proc,
            environment={
                "SLURM_JOB_ID": job_id, "SLURM_STEP_ID": "0",
                "SECRET_TOKEN": "must-not-be-copied",
            },
        )

    def telemetry(self, rank, content):
        directory = self.run / "telemetry"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank-{rank:05d}.jsonl"
        path.write_text(content)
        return path

    def test_startup_records_are_unique_separate_and_do_not_copy_environment(self):
        self.run.mkdir()
        manifest = self.run / "run.json"
        manifest.write_text('{"run_identity_sha256":"unchanged"}')
        first = self.startup()
        second = self.startup()
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists() and second.exists())
        self.assertEqual(first.parent, self.run / "operational/startups")
        payload = json.loads(first.read_text())
        self.assertEqual(payload["schema"], diagnostics.STARTUP_SCHEMA)
        self.assertEqual(payload["job_id"], "495851")
        self.assertEqual(payload["pid"], os.getpid())
        self.assertEqual(payload["rank"], 0)
        self.assertEqual(payload["boot_identity"]["boot_id"], self.boot_id)
        self.assertEqual(payload["boot_identity"]["boot_time_unix"], 1700000000)
        self.assertFalse(payload["scientific_identity_member"])
        self.assertNotIn("must-not-be-copied", first.read_text())
        self.assertEqual(manifest.read_text(), '{"run_identity_sha256":"unchanged"}')
        self.assertEqual(list(first.parent.glob("*.partial")), [])

    def test_unavailable_boot_identity_is_explicit(self):
        result = diagnostics.boot_identity(self.root / "missing-proc")
        self.assertIsNone(result["boot_id"])
        self.assertIsNone(result["boot_time_unix"])
        self.assertEqual(len(result["errors"]), 2)

    def test_invalid_rank_or_job_id_does_not_write(self):
        with self.assertRaises(ValueError):
            self.startup(rank=2, world_size=2)
        with self.assertRaises(ValueError):
            self.startup(job_id="../other-run")
        self.assertFalse(self.run.exists())

    def test_progress_uses_last_complete_record_and_drops_unrelated_payloads(self):
        path = self.telemetry(0, (
            '{"step":2980,"recorded_unix":1700000050,"cumulative_tokens":42,"input_ids":["private"]}\n'
            '{"step":2990,"recorded_unix":'
        ))
        before = path.read_bytes()
        result = diagnostics.last_rank_progress(path, tail_bytes=1024)
        self.assertEqual(result["step"], 2980)
        self.assertEqual(result["cumulative_tokens"], 42)
        self.assertNotIn("private", json.dumps(result))
        self.assertNotIn("input_ids", result)
        self.assertEqual(path.read_bytes(), before)

    def test_tail_is_bounded_and_corruption_does_not_hide_prior_record(self):
        content = "".join(json.dumps({"step": step, "padding": "x" * 40}) + "\n" for step in range(100))
        path = self.telemetry(0, content + "not-json\n")
        result = diagnostics.last_rank_progress(path, tail_bytes=1024)
        self.assertEqual(result["step"], 99)
        self.assertEqual(result["bytes"], len(content + "not-json\n"))
        self.assertEqual(
            diagnostics.last_rank_progress(self.root / "absent")["status"], "unavailable",
        )
        with self.assertRaises(ValueError):
            diagnostics.last_rank_progress(path, tail_bytes=1)

    def test_snapshot_preserves_files_and_identifies_missing_ranks(self):
        self.startup(rank=0, world_size=3)
        self.telemetry(0, '{"step":2990,"recorded_unix":1700000100}\n')
        self.telemetry(1, '{"step":2980,"recorded_unix":1700000070}\n')
        before = {str(path.relative_to(self.run)): path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        result = diagnostics.failure_snapshot(self.run, job_id="495851", include_scheduler=False)
        after = {str(path.relative_to(self.run)): path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(result["expected_world_size"], 3)
        self.assertEqual([row["rank"] for row in result["ranks"]], [0, 1, 2])
        self.assertEqual(result["ranks"][0]["last_progress"]["step"], 2990)
        self.assertEqual(result["ranks"][1]["last_progress"]["step"], 2980)
        self.assertEqual(result["ranks"][2]["last_progress"]["status"], "missing")
        self.assertIsNone(result["ranks"][1]["startup"])
        self.assertTrue(result["ranks"][0]["progress_predates_selected_startup"])
        self.assertFalse(result["scientific_identity_member"])

    def test_startup_records_are_filtered_by_job_and_latest_attempt(self):
        first = self.startup()
        second = self.startup()
        self.startup(rank=1, job_id="123")
        result = diagnostics.failure_snapshot(self.run, job_id="495851", include_scheduler=False)
        self.assertEqual(result["ranks"][0]["startup"]["started_unix_ns"],
                         json.loads(second.read_text())["started_unix_ns"])
        self.assertIsNone(result["ranks"][1]["startup"])
        self.assertTrue(first.exists())

    def test_scheduler_snapshot_is_whitelisted_and_has_bounded_queries(self):
        responses = [
            SimpleNamespace(returncode=0, stdout=(
                "JobId=495851 JobState=FAILED Reason=Node failure NodeList=parrypeak064 "
                "Command=private-command Secret=must-not-leak\n"
            ), stderr=""),
            SimpleNamespace(returncode=0, stdout=(
                "NodeName=parrypeak064 State=IDLE BootTime=2026-09-04T23:46:30 "
                "SlurmdStartTime=2026-09-04T23:48:49 Reason=Unexpected reboot\n"
            ), stderr=""),
        ]
        with mock.patch.object(diagnostics.subprocess, "run", side_effect=responses) as run:
            result = diagnostics.scheduler_snapshot("495851", ["parrypeak064"], timeout_seconds=2)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], "scontrol")
            self.assertEqual(call.kwargs["timeout"], 2)
            self.assertEqual(call.kwargs["env"]["TZ"], "UTC")
            self.assertNotIn("shell", call.kwargs)
        self.assertEqual(result["job"][0]["Reason"], "Node failure")
        self.assertEqual(result["nodes"][0]["Reason"], "Unexpected reboot")
        self.assertNotIn("must-not-leak", json.dumps(result))
        self.assertNotIn("Command", result["job"][0])

    def test_scheduler_timeout_is_reported_without_losing_rank_progress(self):
        self.startup()
        self.telemetry(0, '{"step":2990}\n')
        with mock.patch.object(
            diagnostics.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(["scontrol"], 1),
        ):
            result = diagnostics.failure_snapshot(self.run, job_id="495851", scheduler_timeout=1)
        self.assertEqual(result["ranks"][0]["last_progress"]["step"], 2990)
        self.assertEqual(len(result["scheduler"]["errors"]), 2)
        self.assertEqual(result["scheduler"]["errors"][0]["error"], "TimeoutExpired")

    def test_reboot_detection_compares_operational_boot_time_not_scientific_identity(self):
        path = self.startup()
        startup = json.loads(path.read_text())
        host = startup["hostname"].split(".")[0]
        current_boot = datetime.fromtimestamp(1700000600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        responses = [
            SimpleNamespace(returncode=0, stdout="JobId=495851 JobState=FAILED\n", stderr=""),
            SimpleNamespace(returncode=0, stdout=f"NodeName={host} BootTime={current_boot} State=IDLE\n", stderr=""),
        ]
        with mock.patch.object(diagnostics.subprocess, "run", side_effect=responses):
            snapshot = diagnostics.failure_snapshot(self.run, job_id="495851")
        self.assertTrue(snapshot["ranks"][0]["node_reboot_since_startup"])
        self.assertEqual(snapshot["ranks"][0]["node_boot_time_difference_seconds"], 600)
        self.assertIn("not its OS/hardware/administrative trigger", snapshot["limitations"])

    def test_direct_snapshot_cli_has_no_training_dependencies_and_creates_no_files(self):
        missing = self.root / "not-created"
        process = subprocess.run(
            [sys.executable, "-S", str(MODULE_PATH), "--run-root", str(missing),
             "--job-id", "495851", "--no-scheduler"],
            check=True, capture_output=True, text=True,
        )
        result = json.loads(process.stdout)
        self.assertEqual(result["schema"], diagnostics.SNAPSHOT_SCHEMA)
        self.assertEqual(result["ranks"], [])
        self.assertFalse(missing.exists())

    def test_trainer_startup_call_is_outside_identity_and_before_model_construction(self):
        source = MODULE_PATH.with_name("train.py").read_text()
        tree = ast.parse(source)
        trainer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_train_row_inner")
        calls = [
            node for node in ast.walk(trainer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        startup = [node for node in calls if node.func.id == "record_rank_startup"]
        model = next(node for node in calls if node.func.id == "Metis16ForCausalLM")
        self.assertEqual(len(startup), 1)
        self.assertLess(startup[0].lineno, model.lineno)
        self.assertTrue(any(isinstance(node, ast.Expr) and node.value is startup[0] for node in trainer.body))
        identities = [node for node in calls if node.func.id in {"_run_identity", "_validate_campaign_identity"}]
        for identity in identities:
            self.assertNotIn("boot_identity", ast.unparse(identity))
            self.assertNotIn("record_rank_startup", ast.unparse(identity))


if __name__ == "__main__":
    unittest.main()
