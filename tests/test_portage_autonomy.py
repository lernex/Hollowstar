from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence
from unittest import mock

import yaml

from metis_portage.autotune import (
    Candidate,
    enumerate_performance_candidates,
    load_tuning_bounds,
    tune_family,
    validate_performance_report,
    validate_profile,
)
from metis_portage.config import FamilyTopology, load_portage_config
from metis_portage.discovery import (
    collect_compute_inventory,
    collect_login_inventory,
    require_inventory,
)
from metis_portage.launcher import SlurmSubmitter
from metis_portage.family import (
    FamilySupervisor,
    derive_oom_candidate,
    validate_checkpoint_for_requeue,
    validate_deferred_materialization_request,
    validate_posttraining_state_for_requeue,
    validate_posttraining_batch_migration_for_requeue,
)
from metis_portage.posttraining_release import (
    environment_for_family,
    inspect_posttraining_release_index,
)
from metis_portage.posttraining_builder import _sealed_base_tokenizer
from tests.test_posttraining_release_builder import _base_release
from metis_portage.probes import (
    _MANDATORY_FP8_ROLE_SHAPES,
    _fp8_role_capabilities,
)
from metis_portage.runtime import resolve_runtime
from metis_portage.telemetry import mfu, snapshot_cxi
from metis_portage.util import CommandResult, atomic_write_json, file_sha256, json_sha256
from metis_training.distributed import ParallelTopology
from metis_training.model_config import load_family_config
from metis_training.precision_plan import (
    build_precision_role_plan,
    exact_precision_role_specs,
)
from metis_training.stage_backend import (
    DeferredMaterialization,
    StageBackendError,
    _load_release_index,
    _materialize_generation_hook,
)


_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "metis_portage_login_bootstrap",
    Path(__file__).resolve().parents[1]
    / "ops"
    / "bootstrap-portage-login-runtime.py",
)
assert _BOOTSTRAP_SPEC is not None and _BOOTSTRAP_SPEC.loader is not None
_BOOTSTRAP = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_BOOTSTRAP)


class FakeRunner:
    def __init__(self, responses: Mapping[str, tuple[int, str, str]]) -> None:
        self.responses = dict(responses)
        self.calls: list[list[str]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 60.0,
        cwd=None,
        env=None,
        check: bool = False,
    ) -> CommandResult:
        del timeout, cwd, env
        command = list(argv)
        self.calls.append(command)
        joined = " ".join(command)
        match = next(
            (
                response
                for pattern, response in self.responses.items()
                if pattern in joined
            ),
            (127, "", f"no fake response for {joined}"),
        )
        result = CommandResult(
            argv=tuple(command),
            returncode=match[0],
            stdout=match[1],
            stderr=match[2],
            elapsed_seconds=0.01,
        )
        if check and not result.ok:
            raise RuntimeError(result.stderr)
        return result


class PortageAutonomyTests(unittest.TestCase):
    def test_custom_lustre_root_derives_release_and_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            lustre = Path(raw) / "lustre" / "vollmerc" / "alternate"
            config = load_portage_config(
                environment={"METIS_LUSTRE_ROOT": str(lustre)}
            )
            self.assertEqual(config.lustre_root, lustre.resolve())
            self.assertEqual(
                config.release_root,
                (lustre / "releases" / "metis-1.6-data-r1").resolve(),
            )
            self.assertEqual(
                config.state_root,
                (
                    lustre / "training" / "metis-1.6" / "portage"
                ).resolve(),
            )
            self.assertEqual(
                config.posttraining_release_index,
                (
                    lustre
                    / "releases"
                    / "metis-1.6-posttraining-r1"
                    / "RELEASE_INDEX.json"
                ).resolve(),
            )

    def config(self, temporary: Path, *, posttraining_index: Path | None = None):
        lustre = temporary / "lustre" / "vollmerc" / "metis-1.6"
        release = lustre / "releases" / "metis-1.6-data-r1"
        release.mkdir(parents=True)
        (release / "RELEASE.json").write_text("{}\n", encoding="utf-8")
        environment = {
                "METIS_LUSTRE_ROOT": str(lustre),
                "METIS_DATA_RELEASE": str(release),
                "METIS_PORTAGE_STATE_ROOT": str(lustre / "training" / "portage"),
        }
        if posttraining_index is not None:
            environment["METIS_POSTTRAINING_RELEASE_INDEX"] = str(
                posttraining_index
            )
        return load_portage_config(environment=environment)

    def precision_plan(self, config, family_name: str) -> dict:
        family = next(
            row for row in config.families if row.name == family_name
        )
        model_config = load_family_config(family.manifest)
        measurements = {
            spec.role: {
                "bf16": {
                    "ok": True,
                    "finite_gradients": True,
                    "median_seconds": 2.0,
                },
                "fp8": {
                    "attempted": True,
                    "ok": True,
                    "finite_gradients": True,
                    "median_seconds": 1.0,
                    "loss_relative_error_vs_bf16": 0.01,
                    "error": None,
                },
            }
            for spec in exact_precision_role_specs(model_config)
        }
        return build_precision_role_plan(
            model_config,
            measurements,
            maximum_relative_error=0.03,
        )

    def materialization_fixture(
        self,
        root: Path,
    ) -> tuple[FamilySupervisor, FamilyTopology, dict]:
        config = self.config(root)
        family = config.families[0]
        campaign = config.state_root / "campaign"
        family_root = (
            root / "posttraining-release" / family.name
        ).resolve()
        family_root.mkdir(parents=True)
        executable = family_root / "hooks" / "generate.py"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        executable.chmod(0o750)
        record = {
            "schema": "metis.external-dpd-data/v1",
            "state": "deferred",
            "manifest": "generated/deepseek_dpd_pilot/data/MANIFEST.json",
            "generation_hook": {
                "executable": "hooks/generate.py",
                "executable_sha256": file_sha256(executable),
                "args": ["--sealed"],
                "timeout_seconds": 60,
                "execution": {
                    "protocol": "rank0_only_v1",
                    "nodes": 1,
                    "tasks": 1,
                    "gpus_per_task": 0,
                },
                "receipt": (
                    "generated/deepseek_dpd_pilot/data/REDUCER.json"
                ),
                "rank_receipts": (
                    "generated/deepseek_dpd_pilot/data/ranks"
                ),
            },
        }
        index = {
            "schema": "metis.posttraining-release-index/v1",
            "family": family.name,
            "pipeline_sha256": "1" * 64,
            "tokenizer_manifest": {},
            "requirements": {
                "deepseek_dpd_pilot": {
                    "deepseek_dpd_pilot_data": record
                }
            },
        }
        index["index_sha256"] = json_sha256(index)
        index_path = family_root / "POSTTRAINING_RELEASE.json"
        atomic_write_json(index_path, index)
        deep = {
            "schema": "metis.posttraining-release-deep-verification/v1",
            "complete": True,
        }
        deep["receipt_sha256"] = json_sha256(deep)
        deep_path = root / "posttraining-release" / "DEEP_VERIFICATION.json"
        atomic_write_json(deep_path, deep)
        preflight = {
            "deep_verification": {
                "path": str(deep_path),
                "file_sha256": file_sha256(deep_path),
                "receipt_sha256": deep["receipt_sha256"],
            },
            "family_indexes": {
                family.name: {
                    "path": str(index_path),
                    "file_sha256": file_sha256(index_path),
                    "index_sha256": index["index_sha256"],
                }
            },
        }
        output = campaign / "runs" / family.name
        request_root = (
            output
            / "posttraining"
            / family.name
            / "materialization"
            / "requests"
        )
        checkpoint_root = (
            output
            / "posttraining"
            / family.name
            / "checkpoints"
            / "tokens-0000000000001"
        )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_artifact = checkpoint_root / "rank-00000.pt"
        checkpoint_artifact.write_bytes(b"checkpoint")
        checkpoint_contract = {
            "release_sha256": "1" * 64,
            "shard_manifest_sha256": "2" * 64,
            "family_manifest_sha256": "3" * 64,
            "runtime_manifest_sha256": "4" * 64,
            "autotune_profile_sha256": "5" * 64,
            "precision_role_plan_sha256": "6" * 64,
        }
        checkpoint_manifest = {
            "schema": "metis.distributed-checkpoint/v1",
            "family": family.name,
            "phase": "overall_sft",
            "world_size": family.world_size,
            "expert_parallel_size": family.expert_parallel_size,
            "expert_replica_count": family.expert_replicas,
            **checkpoint_contract,
            "global_token_cursor": 1,
            "optimizer_step": 1,
            "artifacts": [
                {
                    "path": checkpoint_artifact.name,
                    "bytes": checkpoint_artifact.stat().st_size,
                    "sha256": file_sha256(checkpoint_artifact),
                }
            ],
        }
        checkpoint_manifest["checkpoint_sha256"] = json_sha256(
            checkpoint_manifest
        )
        checkpoint_manifest_path = checkpoint_root / "MANIFEST.json"
        atomic_write_json(checkpoint_manifest_path, checkpoint_manifest)
        checkpoint_receipt_path = (
            output
            / "posttraining"
            / family.name
            / "receipts"
            / "overall_sft-checkpoint.json"
        )
        checkpoint_receipt = {
            "schema": "metis.checkpoint-receipt/v1",
            "family": family.name,
            "checkpoint_manifest": str(checkpoint_manifest_path),
            "checkpoint_sha256": checkpoint_manifest[
                "checkpoint_sha256"
            ],
            "precision_role_plan_sha256": checkpoint_contract[
                "precision_role_plan_sha256"
            ],
        }
        checkpoint_receipt["receipt_sha256"] = json_sha256(
            checkpoint_receipt
        )
        atomic_write_json(checkpoint_receipt_path, checkpoint_receipt)
        checkpoint_binding = {
            "stage_id": "overall_sft",
            "checkpoint_path": str(checkpoint_root),
            "checkpoint_sha256": checkpoint_manifest[
                "checkpoint_sha256"
            ],
            "checkpoint_receipt": str(checkpoint_receipt_path),
            "checkpoint_contract": checkpoint_contract,
        }
        request_path = request_root / (
            "deepseek_dpd_pilot--deepseek_dpd_pilot_data--"
            + checkpoint_manifest["checkpoint_sha256"][:16]
            + ".json"
        )
        request = {
            "schema": "metis.deferred-materialization-request/v1",
            "family": family.name,
            "stage": "deepseek_dpd_pilot",
            "requirement": "deepseek_dpd_pilot_data",
            "requirement_schema": record["schema"],
            "parent_checkpoint_sha256": checkpoint_manifest[
                "checkpoint_sha256"
            ],
            "stage_bindings": {
                "parent_policy_checkpoint": checkpoint_binding,
                "dpd_reference_checkpoint": checkpoint_binding,
            },
            "release_index_path": str(index_path),
            "release_index_file_sha256": file_sha256(index_path),
            "release_index_sha256": index["index_sha256"],
            "record_sha256": json_sha256(record),
            "deep_verification": dict(preflight["deep_verification"]),
            "hook": {
                "executable": str(executable),
                "executable_sha256": file_sha256(executable),
                "args": ["--sealed"],
                "timeout_seconds": 60,
                "output_manifest": str(
                    family_root
                    / "generated"
                    / "deepseek_dpd_pilot"
                    / "data"
                    / "MANIFEST.json"
                ),
                "reducer_receipt": str(
                    family_root
                    / "generated"
                    / "deepseek_dpd_pilot"
                    / "data"
                    / "REDUCER.json"
                ),
                "rank_receipts": str(
                    family_root
                    / "generated"
                    / "deepseek_dpd_pilot"
                    / "data"
                    / "ranks"
                ),
                "execution": dict(record["generation_hook"]["execution"]),
                "world_size": 1,
            },
            "trainer_world_size": family.world_size,
            "slurm_job_id": "1234",
            "slurm_restart_count": 0,
            "created_unix": 1,
        }
        request["request_sha256"] = json_sha256(request)
        atomic_write_json(request_path, request)
        supervisor = object.__new__(FamilySupervisor)
        supervisor.config = config
        supervisor.campaign_root = campaign
        supervisor.posttraining_release = preflight
        supervisor.allocated_nodes = []
        supervisor.signal_requested = threading.Event()
        supervisor.processes = []
        supervisor.process_lock = threading.Lock()
        supervisor._environment = lambda _family: {}
        profile_path = (
            campaign / "autotune" / family.name / "profile.json"
        )
        atomic_write_json(profile_path, {"profile": True})
        precision_path = campaign / "precision" / family.name / "plan.json"
        atomic_write_json(precision_path, {"plan": True})
        supervisor.precision_role_plan_paths = {
            family.name: precision_path
        }
        return supervisor, family, request

    def write_materialization_outputs(
        self,
        request: Mapping[str, object],
        *,
        corrupt_rank: bool = False,
    ) -> None:
        hook = request["hook"]
        assert isinstance(hook, Mapping)
        output_path = Path(str(hook["output_manifest"]))
        payload_path = output_path.parent / "payload.bin"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(b"generated")
        output = {
            "envelope_schema": "metis.sealed-artifact/v1",
            "schema": request["requirement_schema"],
            "complete": True,
            "files": [
                {
                    "path": payload_path.name,
                    "bytes": payload_path.stat().st_size,
                    "sha256": file_sha256(payload_path),
                }
            ],
        }
        output["manifest_sha256"] = json_sha256(output)
        atomic_write_json(output_path, output)
        rank_root = Path(str(hook["rank_receipts"]))
        rank_receipt = {
            "schema": "metis.generation-hook-rank-receipt/v1",
            "request_sha256": request["request_sha256"],
            "family": request["family"],
            "stage": request["stage"],
            "requirement": request["requirement"],
            "parent_checkpoint_sha256": request[
                "parent_checkpoint_sha256"
            ],
            "stage_bindings": request["stage_bindings"],
            "rank": 0,
            "world_size": 1,
            "success": not corrupt_rank,
        }
        rank_receipt["receipt_sha256"] = json_sha256(rank_receipt)
        rank_path = rank_root / "rank-00000.json"
        atomic_write_json(rank_path, rank_receipt)
        reducer = {
            "schema": "metis.generation-hook-receipt/v2",
            "request_sha256": request["request_sha256"],
            "family": request["family"],
            "stage": request["stage"],
            "requirement": request["requirement"],
            "parent_checkpoint_sha256": request[
                "parent_checkpoint_sha256"
            ],
            "stage_bindings": request["stage_bindings"],
            "release_index_file_sha256": request[
                "release_index_file_sha256"
            ],
            "release_index_sha256": request["release_index_sha256"],
            "record_sha256": request["record_sha256"],
            "deep_verification_file_sha256": request[
                "deep_verification"
            ]["file_sha256"],
            "deep_verification_receipt_sha256": request[
                "deep_verification"
            ]["receipt_sha256"],
            "executable_sha256": hook["executable_sha256"],
            "execution_protocol": hook["execution"]["protocol"],
            "world_size": 1,
            "output_manifest_sha256": file_sha256(output_path),
            "output_manifest_self_sha256": output["manifest_sha256"],
            "success": True,
            "rank_receipts": [
                {
                    "rank": 0,
                    "file_sha256": file_sha256(rank_path),
                    "receipt_sha256": rank_receipt["receipt_sha256"],
                }
            ],
        }
        reducer["receipt_sha256"] = json_sha256(reducer)
        atomic_write_json(Path(str(hook["reducer_receipt"])), reducer)

    def test_locked_family_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            topologies = {
                family.name: (
                    family.nodes,
                    family.world_size,
                    family.expert_parallel_size,
                    family.expert_replicas,
                )
                for family in config.families
            }
            self.assertEqual(topologies["praxis"], (40, 160, 32, 5))
            self.assertEqual(topologies["logos"], (88, 352, 32, 11))

    def test_login_bootstrap_accepts_only_self_hashed_pinned_pyyaml_wheel(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "PyYAML-6.0.2-py3-none-any.whl"
            wheel.write_bytes(b"offline pinned wheel fixture")
            bundle = {
                "schema": "metis.portage-runtime-bundle/v1",
                "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                "torch_version": "2.9.0",
                "rocm_version": "7.0",
                "wheels": [
                    {
                        "distribution": "PyYAML",
                        "version": "6.0.2",
                        "path": wheel.name,
                        "sha256": file_sha256(wheel),
                    }
                ],
                "sources": [],
            }
            bundle["bundle_sha256"] = json_sha256(bundle)
            path = root / "runtime-bundle.json"
            atomic_write_json(path, bundle)
            loaded, pinned = _BOOTSTRAP.validate_login_bundle(
                path,
                python_abi=bundle["python_abi"],
            )
            self.assertEqual(loaded["bundle_sha256"], bundle["bundle_sha256"])
            self.assertEqual(pinned["sha256"], file_sha256(wheel))
            wheel.write_bytes(b"same trust boundary, changed bytes")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                _BOOTSTRAP.validate_login_bundle(
                    path,
                    python_abi=bundle["python_abi"],
                )

    def test_login_inventory_passes_only_live_portage_lustre_and_gpu_tres(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            runner = FakeRunner(
                {
                    "scontrol show config": (0, "ClusterName = portage\n", ""),
                    "show partition": (
                        0,
                        "PartitionName=parry TotalNodes=128 MaxTime=5-00:00:00 "
                        "TRES=cpu=3072,gres/gpu=512\n",
                        "",
                    ),
                    "show nodes": (
                        0,
                        "NodeName=p001 CfgTRES=cpu=24,mem=512000M,gres/gpu=4 Gres=gpu:mi300a:4\n",
                        "",
                    ),
                    "sinfo ": (0, "parry|128|gpu:mi300a:4|5-00:00:00|up|128/0/0/128\n", ""),
                    "findmnt": (0, f"lustre fs {config.release_root}\n", ""),
                    "lfs getstripe": (0, "stripe_count: 16\n", ""),
                    "rev-parse HEAD": (0, "a" * 40 + "\n", ""),
                    "status --porcelain": (0, "", ""),
                    "sbatch --help": (
                        0,
                        "--gpus-per-node --signal --test-only\n",
                        "",
                    ),
                    "sbatch --test-only": (
                        0,
                        "sbatch: Job 12345 to start at 2026-07-25T00:00:00\n",
                        "",
                    ),
                    "srun --help": (0, "--relative --exact\n", ""),
                    "sacct --help": (0, "sacct\n", ""),
                    "import json,platform,sys": (0, '{"executable":"python3"}\n', ""),
                    "import json,torch": (
                        0,
                        '{"torch":"2.9","hip":"7.0","cuda":null,"cuda_available":false}\n',
                        "",
                    ),
                    "module -t list": (0, "rocm/7.0:pytorch/2.9\n", ""),
                }
            )
            record = collect_login_inventory(config, runner=runner)
            require_inventory(record)
            self.assertEqual(record["facts"]["gpu_count_per_node"], 4)
            self.assertEqual(record["facts"]["partition_nodes"], 128)
            submission_probe = next(
                call
                for call in runner.calls
                if call[:2] == ["sbatch", "--test-only"]
            )
            self.assertIn("--nodes", submission_probe)
            self.assertEqual(
                submission_probe[submission_probe.index("--nodes") + 1],
                "128",
            )
            self.assertEqual(
                submission_probe[submission_probe.index("--ntasks") + 1],
                "512",
            )

    def test_login_inventory_fails_closed_on_wrong_gpu_tres(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            runner = FakeRunner(
                {
                    "scontrol show config": (0, "ClusterName = portage\n", ""),
                    "show partition": (
                        0,
                        "PartitionName=parry TotalNodes=128 MaxTime=5-00:00:00\n",
                        "",
                    ),
                    "show nodes": (0, "NodeName=p001 CfgTRES=gres/gpu=2 Gres=gpu:2\n", ""),
                    "sinfo ": (0, "parry\n", ""),
                    "findmnt": (0, f"lustre fs {config.release_root}\n", ""),
                    "lfs getstripe": (0, "", ""),
                    "rev-parse HEAD": (0, "a" * 40, ""),
                    "status --porcelain": (0, "", ""),
                    "sbatch --help": (0, "ok", ""),
                    "sbatch --test-only": (1, "", "invalid association"),
                    "srun --help": (0, "ok", ""),
                    "sacct --help": (0, "ok", ""),
                    "import json,platform,sys": (0, "{}", ""),
                    "import json,torch": (0, "{}", ""),
                    "module -t list": (0, "", ""),
                }
            )
            record = collect_login_inventory(config, runner=runner)
            with self.assertRaisesRegex(RuntimeError, "gpu-tres"):
                require_inventory(record)

    def test_compute_inventory_requires_rocm_gfx942_hbm_rccl_and_cxi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            torch_row = {
                "torch": "2.9.0",
                "hip": "7.0",
                "cuda": None,
                "distributed_available": True,
                "nccl_available": True,
                "device_count": 1,
                "device_name": "AMD Instinct MI300A",
                "total_memory": 128_000_000_000,
                "gcn_arch_name": "gfx942:sramecc+:xnack-",
                "bf16_supported": True,
                "float8_dtypes": ["float8_e4m3fnuz"],
            }
            runner = FakeRunner(
                {
                    "rocminfo": (0, "Name: gfx942\n", ""),
                    "rocm-smi": (0, '{"card0":{}}\n', ""),
                    "hipconfig": (0, "HIP version: 7.0\n", ""),
                    "fi_info": (0, "provider: cxi\n", ""),
                    "row.update": (0, json.dumps(torch_row) + "\n", ""),
                    "torch.cuda.nccl.version": (0, '{"nccl_version":[2,22,3]}\n', ""),
                    "findmnt -t lustre": (0, "lustre /lus/lustre1\n", ""),
                    "numactl": (0, "available: 4 nodes\n", ""),
                    "lspci": (0, "Display controller: AMD\n", ""),
                    "readable_files": (
                        0,
                        '{"exists":true,"readable_files":12,"sample":["/run/cxi/cxi0/foo"]}\n',
                        "",
                    ),
                }
            )
            record = collect_compute_inventory(config, runner=runner)
            require_inventory(record)
            self.assertEqual(record["facts"]["gpu_arch"], "gfx942:sramecc+:xnack-")

    def _manifest(self, path: Path) -> Path:
        payload = {
            "schema": "test",
            "autotune": {
                "bounds": {
                    "micro_batch_sizes": [4, 2, 1],
                    "grad_accum_steps": [1, 2, 4, 8],
                    "global_token_batch": {
                        "min": 524288,
                        "target": 2097152,
                        "max": 8388608,
                    },
                    "learning_rates": [0.0002, 0.0003, 0.0004],
                    "preferred_learning_rate": 0.0003,
                    "compile_modes": ["max-autotune", "eager"],
                    "precision_profiles": ["fp8", "bf16"],
                    "dispatch_overlap": [True, False],
                    "ngram_table_modes": ["replicated", "row_sharded"],
                    "gates": {
                        "max_hbm_fraction": 0.88,
                        "max_fp8_loss_relative_error": 0.03,
                        "max_ngram_layout_loss_relative_error": 0.001,
                        "max_update_to_weight_ratio": 0.01,
                        "max_grad_norm": 5.0,
                    },
                }
            },
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_autotune_candidates_are_manifest_bounded_and_fp8_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory) / "praxis.yaml")
            bounds = load_tuning_bounds(
                manifest,
                default_maximum_hbm_fraction=0.88,
            )
            rows = enumerate_performance_candidates(
                bounds,
                world_size=128,
                sequence_length=4096,
                maximum_trials=24,
            )
            self.assertTrue(rows)
            self.assertEqual(rows[0].precision_profile, "fp8")
            fp8_rows = [
                row for row in rows if row.precision_profile == "fp8"
            ]
            self.assertEqual(
                {row.compile_mode for row in fp8_rows},
                set(bounds.compile_modes),
            )
            self.assertEqual(
                {row.dispatch_overlap for row in fp8_rows},
                set(bounds.dispatch_overlap),
            )
            self.assertEqual(
                {row.ngram_table_mode for row in fp8_rows},
                set(bounds.ngram_table_modes),
            )
            self.assertEqual(
                {row.micro_batch_size for row in fp8_rows},
                set(bounds.micro_batch_sizes),
            )
            for row in rows:
                global_tokens = (
                    row.micro_batch_size * row.grad_accum_steps * 128 * 4096
                )
                self.assertGreaterEqual(global_tokens, bounds.global_token_batch_min)
                self.assertLessEqual(global_tokens, bounds.global_token_batch_max)

    def test_probe_report_rejects_token_drop_and_hbm_overcommit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bounds = load_tuning_bounds(
                self._manifest(Path(directory) / "praxis.yaml"),
                default_maximum_hbm_fraction=0.88,
            )
            candidate = Candidate(1, 4, "fp8", "eager", True)
            report = {
                "ok": True,
                "finite_loss": True,
                "step_time_s": 1.0,
                "non_padding_tokens": 1000,
                "estimated_train_flops": 100000,
                "peak_hbm_bytes": 120,
                "overflow_drop_tokens": 1,
                "collective_errors": 0,
                "loss_relative_error_vs_bf16": 0.01,
                "ngram_table_mode": candidate.ngram_table_mode,
            }
            passed, reasons, _ = validate_performance_report(
                report,
                candidate=candidate,
                bounds=bounds,
                hbm_bytes=128,
            )
            self.assertFalse(passed)
            self.assertIn("MoE overflow dropped tokens", reasons)
            self.assertIn("HBM headroom gate failed", reasons)

    def test_single_apu_fp8_gate_requires_every_exact_role_shape(self) -> None:
        measurements = [
            {
                "m": shape[0],
                "k": shape[1],
                "n": shape[2],
                "tflops": 100.0,
                "relative_l2_error": 0.01,
            }
            for _role, shape in _MANDATORY_FP8_ROLE_SHAPES[:-1]
        ]
        capabilities = _fp8_role_capabilities(
            measurements,
            [],
            maximum_error=0.03,
        )
        self.assertEqual(
            set(capabilities),
            {role for role, _shape in _MANDATORY_FP8_ROLE_SHAPES},
        )
        self.assertFalse(
            capabilities[_MANDATORY_FP8_ROLE_SHAPES[-1][0]]["ok"]
        )
        self.assertFalse(all(row["ok"] for row in capabilities.values()))
        role, shape = _MANDATORY_FP8_ROLE_SHAPES[-1]
        measurements.append(
            {
                "m": shape[0],
                "k": shape[1],
                "n": shape[2],
                "tflops": 100.0,
                "relative_l2_error": 0.01,
            }
        )
        complete = _fp8_role_capabilities(
            measurements,
            [],
            maximum_error=0.03,
        )
        self.assertTrue(complete[role]["ok"])
        self.assertTrue(all(row["ok"] for row in complete.values()))

    def test_tuner_hashes_profile_and_prefers_stable_manifest_lr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = self.config(temporary)
            manifest = self._manifest(temporary / "praxis.yaml")
            family = FamilyTopology(
                name="praxis",
                nodes=32,
                world_size=128,
                relative_node=0,
                expert_parallel_size=128,
                expert_replicas=1,
                manifest=manifest,
            )
            role_plan = self.precision_plan(config, "praxis")
            performance_calls: list[Candidate] = []

            def trial(candidate: Candidate, report_path: Path, optimizer: bool):
                if optimizer:
                    report = {
                        "ok": True,
                        "finite_loss": True,
                        "initial_loss": 5.0,
                        "final_loss": 4.0 if candidate.learning_rate == 0.0002 else 4.2,
                        "max_grad_norm": 1.0,
                        "nonfinite_steps": 0,
                        "update_to_weight_ratio": 0.001,
                        "precision_role_plan_sha256": role_plan["plan_sha256"],
                    }
                else:
                    performance_calls.append(candidate)
                    report = {
                        "ok": True,
                        "finite_loss": True,
                        "step_time_s": (
                            (2.0 if candidate.precision_profile == "fp8" else 0.5)
                            * (
                                0.9
                                if candidate.ngram_table_mode == "row_sharded"
                                else 1.0
                            )
                        ),
                        "non_padding_tokens": 1_000_000,
                        "estimated_train_flops": 1e15,
                        "peak_hbm_bytes": 80_000_000_000,
                        "overflow_drop_tokens": 0,
                        "collective_errors": 0,
                        "loss_relative_error_vs_bf16": 0.01,
                        "ngram_table_mode": candidate.ngram_table_mode,
                        "ngram_layout_loss_relative_error": 0.0,
                        "final_loss": 4.5,
                        "precision_role_plan_sha256": role_plan["plan_sha256"],
                    }
                atomic_write_json(report_path, report)
                return report

            marker = {"marker_sha256": "b" * 64}
            profile = tune_family(
                config=config,
                family=family,
                inventory_fingerprint="a" * 64,
                release_marker=marker,
                hbm_bytes=128_000_000_000,
                output_directory=temporary / "tune",
                run_trial=trial,
                precision_role_plan=role_plan,
            )
            self.assertEqual(profile["selected"]["precision_profile"], "fp8")
            self.assertEqual(profile["selected"]["learning_rate"], 0.0003)
            self.assertEqual(
                profile["selected"]["ngram_table_mode"],
                "row_sharded",
            )
            self.assertEqual(profile["performance_coverage"]["missing"], {})
            self.assertLessEqual(
                profile["performance_coverage"]["performance_trials_used"],
                config.raw["autotune"]["maximum_trials_per_family"],
            )
            self.assertEqual(
                profile["performance_coverage"]["performance_trials_used"],
                len(performance_calls),
            )
            self.assertTrue(
                any(
                    row["kind"] == "ngram_replicated_reference"
                    for row in profile["trials"]
                )
            )
            bf16_rows = [
                row
                for row in profile["trials"]
                if row["candidate"]["precision_profile"] == "bf16"
                and row["passed"]
            ]
            self.assertTrue(bf16_rows)
            self.assertGreater(
                max(row["tokens_per_second"] for row in bf16_rows),
                profile["measured_tokens_per_second"],
            )
            validated = validate_profile(
                temporary / "tune" / "profile.json",
                family=family,
                inventory_fingerprint="a" * 64,
                release_marker_sha256="b" * 64,
            )
            self.assertEqual(
                validated["profile_sha256"],
                json_sha256(validated, omit=("profile_sha256",)),
            )

    def test_tuner_refuses_selection_without_safe_fp8_axis_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = self.config(temporary)
            manifest = self._manifest(temporary / "praxis.yaml")
            family = FamilyTopology(
                name="praxis",
                nodes=32,
                world_size=128,
                relative_node=0,
                expert_parallel_size=128,
                expert_replicas=1,
                manifest=manifest,
            )
            role_plan = self.precision_plan(config, "praxis")

            def trial(candidate: Candidate, report_path: Path, optimizer: bool):
                self.assertFalse(optimizer)
                safe = candidate.compile_mode != "eager"
                report = {
                    "ok": safe,
                    "finite_loss": safe,
                    "step_time_s": 1.0,
                    "non_padding_tokens": 1_000_000,
                    "estimated_train_flops": 1e15,
                    "peak_hbm_bytes": 80_000_000_000,
                    "overflow_drop_tokens": 0,
                    "collective_errors": 0,
                    "loss_relative_error_vs_bf16": 0.01,
                    "ngram_table_mode": candidate.ngram_table_mode,
                    "ngram_layout_loss_relative_error": 0.0,
                    "final_loss": 4.5,
                    "precision_role_plan_sha256": role_plan["plan_sha256"],
                }
                atomic_write_json(report_path, report)
                return report

            with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
                tune_family(
                    config=config,
                    family=family,
                    inventory_fingerprint="a" * 64,
                    release_marker={"marker_sha256": "b" * 64},
                    hbm_bytes=128_000_000_000,
                    output_directory=temporary / "tune-incomplete",
                    run_trial=trial,
                    precision_role_plan=role_plan,
                )

    def test_runtime_selects_only_measured_rocm_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = self.config(temporary)
            runner = FakeRunner(
                {
                    "module load pytorch/2.9-rocm": (
                        0,
                        json.dumps(
                            {
                                "marker": "metis_runtime_probe",
                                "python": {
                                    "implementation": "cpython",
                                    "version": "3.11.9",
                                    "abi": "cp311",
                                    "soabi": "cpython-311-x86_64-linux-gnu",
                                    "executable": "python3",
                                },
                                "platform": {
                                    "system": "Linux",
                                    "machine": "x86_64",
                                    "libc": ["glibc", "2.38"],
                                },
                                "packages": {
                                    "PyYAML": {"ok": True, "version": "6.0.2"},
                                    "numpy": {"ok": True, "version": "2.1.0"},
                                    "tokenizers": {"ok": True, "version": "0.21.0"},
                                    "safetensors": {"ok": True, "version": "0.5.0"},
                                    "torch": {"ok": True, "version": "2.9.0"},
                                    "triton": {"ok": True, "version": "3.2.0"},
                                    "transformer-engine": {
                                        "ok": False,
                                        "version": None,
                                    },
                                    "mamba-ssm": {"ok": True, "version": "2.2.4"},
                                    "causal-conv1d": {
                                        "ok": True,
                                        "version": "1.4.0",
                                    },
                                    "aiter": {"ok": False, "version": None},
                                    "flash-attn": {
                                        "ok": True,
                                        "version": "2.8.0",
                                    },
                                },
                                "torch": {
                                    "ok": True,
                                    "version": "2.9.0",
                                    "hip": "7.0",
                                    "cuda": None,
                                    "git_version": "abc",
                                    "cxx11_abi": True,
                                    "cuda_available": False,
                                    "device_count": 0,
                                },
                            }
                        )
                        + "\n",
                        "",
                    ),
                    "metis_runtime_probe": (
                        0,
                        json.dumps(
                            {
                                "marker": "metis_runtime_probe",
                                "python": {
                                    "implementation": "cpython",
                                    "version": "3.11.9",
                                    "abi": "cp311",
                                },
                                "packages": {},
                                "torch": {
                                    "ok": True,
                                    "version": "2.9.0",
                                    "hip": None,
                                    "cuda": None,
                                },
                            }
                        )
                        + "\n",
                        "",
                    ),
                    # Active interpreter is not a ROCm build.
                    "module -t avail": (
                        0,
                        "rocm/7.0\npytorch/2.9-rocm\n",
                        "",
                    ),
                }
            )
            record = resolve_runtime(
                config,
                output_directory=temporary / "runtime",
                runner=runner,
            )
            self.assertEqual(record["modules"], ["pytorch/2.9-rocm"])
            self.assertIn(
                "module load pytorch/2.9-rocm",
                (temporary / "runtime" / "runtime.sh").read_text(encoding="utf-8"),
            )

    def test_slurm_plan_preserves_dependency_and_full_family_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = self.config(temporary)
            campaign = config.state_root / "campaigns" / "abc"
            (campaign / "logs").mkdir(parents=True)
            runtime = campaign / "runtime.sh"
            runtime.write_text("#!/bin/bash\n", encoding="utf-8")
            runner = FakeRunner(
                {
                    "portage-stage.sbatch": (0, "101;portage\n", ""),
                    "portage-family.sbatch": (0, "102;portage\n", ""),
                }
            )
            submitter = SlurmSubmitter(
                config,
                campaign,
                runtime,
                runner=runner,
            )
            stage_id = submitter.submit_stage(
                {
                    "id": "single_apu",
                    "nodes": 1,
                    "tasks": 1,
                    "gpus_per_node": 1,
                    "time": "00:20:00",
                },
                None,
            )
            family_id = submitter.submit_family(stage_id)
            self.assertEqual((stage_id, family_id), ("101", "102"))
            family_call = runner.calls[-1]
            self.assertIn("128", family_call)
            self.assertIn("512", family_call)
            self.assertIn("afterok:101", family_call)
            self.assertIn("--signal=B:USR1@900", family_call)

    def test_oom_revision_preserves_checkpoint_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            family = config.families[0]
            profile = {
                "selected": {
                    "micro_batch_size": 4,
                    "grad_accum_steps": 1,
                    "precision_profile": "fp8",
                    "compile_mode": "default",
                    "dispatch_overlap": True,
                    "ngram_table_mode": "row_sharded",
                    "learning_rate": 0.00018,
                }
            }
            candidate = derive_oom_candidate(
                profile=profile,
                family=family,
                config=config,
            )
            self.assertEqual(
                (
                    candidate.micro_batch_size,
                    candidate.grad_accum_steps,
                ),
                (2, 2),
            )
            self.assertEqual(
                candidate.micro_batch_size * candidate.grad_accum_steps,
                4,
            )
            self.assertEqual(candidate.precision_profile, "fp8")
            self.assertEqual(candidate.ngram_table_mode, "row_sharded")

    def test_topology_race_receipt_has_one_canonical_self_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            supervisor = object.__new__(FamilySupervisor)
            supervisor.campaign_root = root / "campaign"
            supervisor.config = config
            supervisor.fingerprint = "f" * 64
            supervisor.hbm_bytes = 1
            profiles = {
                family.name: {
                    "family": family.name,
                    "selected": {
                        "micro_batch_size": 1,
                        "grad_accum_steps": 1,
                        "precision_profile": "fp8",
                        "compile_mode": "default",
                        "dispatch_overlap": True,
                        "ngram_table_mode": "row_sharded",
                    },
                    "profile_sha256": "0" * 64,
                }
                for family in config.families
            }

            def runner_for(_family, *, placement="contiguous"):
                def run(_candidate, output, _optimizer):
                    report = {"placement": placement}
                    atomic_write_json(output, report)
                    return report

                return run
            supervisor.trial_runner = runner_for
            with mock.patch(
                "metis_portage.family.load_tuning_bounds",
                return_value=object(),
            ), mock.patch(
                "metis_portage.family.validate_performance_report",
                side_effect=lambda report, **_kwargs: (
                    True,
                    [],
                    2.0 if report["placement"] == "contiguous" else 1.0,
                ),
            ):
                supervisor.topology_race(profiles)
            race = json.loads(
                (
                    supervisor.campaign_root
                    / "autotune"
                    / "topology"
                    / "race.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                race["race_sha256"],
                json_sha256(race, omit=("race_sha256",)),
            )

    def test_training_telemetry_averages_repeated_global_throughput_and_rejects_rank_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            family = SimpleNamespace(name="praxis", world_size=2)
            supervisor = object.__new__(FamilySupervisor)
            supervisor.campaign_root = root
            supervisor.config = SimpleNamespace(
                families=(family,),
                raw={"training": {"output_subdirectory": "training"}},
            )
            telemetry = root / "training" / "praxis" / "telemetry"
            telemetry.mkdir(parents=True)

            def write(rank: int, throughput: float) -> None:
                row = {
                    "tokens_per_second": throughput,
                    "estimated_train_flops": 1.0,
                    "step_time_s": 1.0,
                    "mfu": 0.5,
                    "all_to_all_bytes": 1.0,
                    "all_to_all_seconds": 1.0,
                    "overflow_drop_tokens": 0,
                    "expert_load_cv": 0.1,
                    "loss": 1.0,
                    "global_token_cursor": 1_000_000_000_000,
                }
                (telemetry / f"rank-{rank:05d}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )

            write(0, 100.0)
            write(1, 104.0)
            summary = supervisor.validate_training_telemetry(
                {"praxis": {"profile_sha256": "a" * 64}}
            )
            family_summary = summary["families"]["praxis"]
            self.assertAlmostEqual(
                family_summary["tokens_per_second_global_mean"], 102.0
            )
            self.assertNotIn("tokens_per_second_sum", family_summary)

            write(1, 200.0)
            with self.assertRaisesRegex(
                RuntimeError, "disagree on global tokens_per_second"
            ):
                supervisor.validate_training_telemetry(
                    {"praxis": {"profile_sha256": "a" * 64}}
                )

    def test_supervisor_materializes_after_trainer_exit_then_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor, family, request = self.materialization_fixture(root)
            trainer_calls: list[str] = []

            def train(_family, _profile):
                trainer_calls.append("train")
                return 252 if len(trainer_calls) == 1 else 0

            launched: list[list[str]] = []

            def run(argv, **_kwargs):
                launched.append(list(argv))
                self.write_materialization_outputs(request)
                return 0

            supervisor.train = train
            supervisor._run = run
            with mock.patch.dict(
                "os.environ",
                {"SLURM_JOB_ID": "1234", "SLURM_RESTART_COUNT": "0"},
                clear=False,
            ):
                returncode = supervisor.train_with_deferred_materialization(
                    family,
                    {"topology_placement": "contiguous"},
                )
            self.assertEqual(returncode, 0)
            self.assertEqual(trainer_calls, ["train", "train"])
            self.assertEqual(len(launched), 1)
            self.assertEqual(launched[0][0], "srun")
            self.assertEqual(launched[0].count("srun"), 1)
            completions = list(
                (
                    supervisor._output_for(family)
                    / "posttraining"
                    / family.name
                    / "materialization"
                    / "completions"
                ).glob("*.json")
            )
            self.assertEqual(len(completions), 1)
            completion = json.loads(completions[0].read_text(encoding="utf-8"))
            self.assertTrue(completion["resume_authorized"])
            self.assertEqual(
                completion["completion_sha256"],
                json_sha256(completion, omit=("completion_sha256",)),
            )

    def test_deferred_request_tamper_and_nested_srun_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor, family, request = self.materialization_fixture(root)
            request_path = next(
                (
                    supervisor._output_for(family)
                    / "posttraining"
                    / family.name
                    / "materialization"
                    / "requests"
                ).glob("*.json")
            )
            tampered = dict(request)
            tampered["stage"] = "dpd"
            atomic_write_json(request_path, tampered)
            with self.assertRaisesRegex(RuntimeError, "lineage is invalid"):
                validate_deferred_materialization_request(
                    request_path,
                    output=supervisor._output_for(family),
                    family=family,
                    posttraining_preflight=supervisor.posttraining_release,
                    expected_job_id="1234",
                    expected_restart_count=0,
                )
            atomic_write_json(request_path, request)
            with mock.patch.dict(
                "os.environ",
                {
                    "SLURM_JOB_ID": "1234",
                    "SLURM_RESTART_COUNT": "0",
                    "SLURM_STEP_ID": "7",
                },
                clear=False,
            ), self.assertRaisesRegex(RuntimeError, "inside another Slurm step"):
                supervisor.materialize_deferred_requirement(
                    family=family,
                    profile={"topology_placement": "contiguous"},
                    restart_count=0,
                )

    def test_corrupt_rank_receipts_retry_to_cap_then_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor, family, request = self.materialization_fixture(root)
            calls = 0

            def run(_argv, **_kwargs):
                nonlocal calls
                calls += 1
                self.write_materialization_outputs(
                    request,
                    corrupt_rank=True,
                )
                return 0

            supervisor._run = run
            with mock.patch.dict(
                "os.environ",
                {"SLURM_JOB_ID": "1234", "SLURM_RESTART_COUNT": "0"},
                clear=False,
            ), self.assertRaisesRegex(RuntimeError, "exhausted 3 attempts"):
                supervisor.materialize_deferred_requirement(
                    family=family,
                    profile={"topology_placement": "contiguous"},
                    restart_count=0,
                )
            self.assertEqual(calls, 3)

    def test_early_signal_writes_valid_restart_from_origin_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor = object.__new__(FamilySupervisor)
            supervisor.config = self.config(root)
            supervisor.campaign_root = (
                supervisor.config.state_root / "early-signal"
            )
            supervisor.campaign_root.mkdir(parents=True)
            supervisor.signal_requested = threading.Event()
            supervisor.processes = []
            supervisor.process_lock = threading.Lock()

            def audit(_family):
                supervisor.signal_requested.set()
                return {}

            supervisor.audit = audit
            with mock.patch.dict("os.environ", {}, clear=False):
                returncode = supervisor.run()
            self.assertEqual(returncode, 75)
            marker = json.loads(
                (supervisor.campaign_root / "requeue.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                marker["marker_sha256"],
                json_sha256(marker, omit=("marker_sha256",)),
            )
            self.assertEqual(marker["classification"], "signal_during_audit")
            self.assertTrue(
                all(
                    row["status"] == "restart_from_origin"
                    for row in marker["checkpoints"].values()
                )
            )
            self.assertTrue(
                all(
                    row["status"] == "not_started"
                    for row in marker["posttraining"].values()
                )
            )

    def test_posttraining_requeue_binds_active_checkpoint_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            family = config.families[0]
            output = root / "output"
            post_root = output / "posttraining" / family.name
            checkpoint = (
                post_root / "checkpoints" / "tokens-1000000004096"
            )
            checkpoint.mkdir(parents=True)
            artifact = checkpoint / "replicated.pt"
            artifact.write_bytes(b"posttraining-state")
            contract = {
                "release_sha256": "1" * 64,
                "shard_manifest_sha256": "2" * 64,
                "family_manifest_sha256": "3" * 64,
                "runtime_manifest_sha256": "4" * 64,
                "autotune_profile_sha256": "5" * 64,
                "precision_role_plan_sha256": "6" * 64,
            }
            active = {
                "kind": "policy",
                "stage_id": "context_extension",
                "stage_config_sha256": "7" * 64,
                "parent_checkpoint_sha256": "8" * 64,
                "bundle_sha256": "2" * 64,
                "optimizer_state_policy": "preserve",
                "runtime_batch": {
                    "micro_batch_size": 1,
                    "gradient_accumulation": 1,
                },
                "epoch": 0,
                "next_global_batch": 1,
                "optimizer_step": 1,
                "campaign_token_cursor": 1_000_000_004_096,
                "checkpoint_path": str(checkpoint),
                "checkpoint_contract": contract,
            }
            manifest = {
                "schema": "metis.distributed-checkpoint/v1",
                "family": family.name,
                "world_size": family.world_size,
                "expert_parallel_size": family.expert_parallel_size,
                "expert_replica_count": family.expert_replicas,
                "global_token_cursor": active["campaign_token_cursor"],
                "optimizer_step": active["optimizer_step"],
                "phase": active["stage_id"],
                **contract,
                "extra_state": {
                    "posttraining_stage": active["stage_id"],
                    "parent_checkpoint_sha256": active[
                        "parent_checkpoint_sha256"
                    ],
                    "bundle_sha256": active["bundle_sha256"],
                    "stage_epoch": active["epoch"],
                    "stage_next_global_batch": active[
                        "next_global_batch"
                    ],
                    "stage_optimizer_step": active["optimizer_step"],
                    "campaign_token_cursor": active[
                        "campaign_token_cursor"
                    ],
                    "runtime_batch": active["runtime_batch"],
                },
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": file_sha256(artifact),
                    }
                ],
            }
            manifest["checkpoint_sha256"] = json_sha256(manifest)
            active["checkpoint_sha256"] = manifest["checkpoint_sha256"]
            atomic_write_json(checkpoint / "MANIFEST.json", manifest)
            state = {
                "schema": "metis.inprocess-posttraining-state/v1",
                "family": family.name,
                "pipeline_sha256": "9" * 64,
                "family_manifest_sha256": "3" * 64,
                "base_checkpoint_sha256": "8" * 64,
                "policy_checkpoint_sha256": "8" * 64,
                "policy_checkpoint_path": str(root / "base"),
                "policy_checkpoint_receipt": str(root / "base-receipt.json"),
                "policy_checkpoint_contract": None,
                "campaign_token_cursor": active[
                    "campaign_token_cursor"
                ],
                "reward_model_manifest": None,
                "evaluation_receipt": None,
                "completed": [],
                "active": active,
            }
            state["state_sha256"] = json_sha256(state)
            atomic_write_json(post_root / "STATE.json", state)
            observed = validate_posttraining_state_for_requeue(
                output,
                family=family,
            )
            self.assertEqual(observed["status"], "active")
            self.assertEqual(
                observed["active"]["checkpoint"]["checkpoint_sha256"],
                manifest["checkpoint_sha256"],
            )

            stage_metrics = {"loss": 1.0}
            stage_receipt = {
                "schema": "metis.inprocess-stage-receipt/v1",
                "family": family.name,
                "stage": "context_extension",
                "precision_role_plan_sha256": contract[
                    "precision_role_plan_sha256"
                ],
                "metrics": stage_metrics,
            }
            stage_receipt["receipt_sha256"] = json_sha256(stage_receipt)
            stage_receipt_path = (
                post_root / "receipts" / "context_extension-output.json"
            )
            atomic_write_json(stage_receipt_path, stage_receipt)
            checkpoint_receipt = {
                "schema": "metis.checkpoint-receipt/v1",
                "family": family.name,
                "stage": "context_extension",
                "checkpoint_manifest": str(
                    (checkpoint / "MANIFEST.json").resolve()
                ),
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "precision_role_plan_sha256": contract[
                    "precision_role_plan_sha256"
                ],
            }
            checkpoint_receipt["receipt_sha256"] = json_sha256(
                checkpoint_receipt
            )
            checkpoint_receipt_path = (
                post_root
                / "receipts"
                / "context_extension-checkpoint.json"
            )
            atomic_write_json(checkpoint_receipt_path, checkpoint_receipt)
            state["completed"] = [
                {
                    "stage_id": "context_extension",
                    "output_receipt": str(stage_receipt_path),
                    "output_receipt_sha256": stage_receipt[
                        "receipt_sha256"
                    ],
                    "metrics_sha256": json_sha256(stage_metrics),
                }
            ]
            state["active"] = None
            state["policy_checkpoint_sha256"] = manifest[
                "checkpoint_sha256"
            ]
            state["policy_checkpoint_path"] = str(checkpoint)
            state["policy_checkpoint_receipt"] = str(
                checkpoint_receipt_path
            )
            state["policy_checkpoint_contract"] = contract
            state["state_sha256"] = json_sha256(
                state, omit=("state_sha256",)
            )
            atomic_write_json(post_root / "STATE.json", state)
            boundary = validate_posttraining_state_for_requeue(
                output,
                family=family,
            )
            self.assertEqual(boundary["status"], "stage_boundary")
            checkpoint_receipt["checkpoint_sha256"] = "f" * 64
            checkpoint_receipt["receipt_sha256"] = json_sha256(
                checkpoint_receipt, omit=("receipt_sha256",)
            )
            atomic_write_json(checkpoint_receipt_path, checkpoint_receipt)
            with self.assertRaisesRegex(
                RuntimeError, "checkpoint receipt is invalid"
            ):
                validate_posttraining_state_for_requeue(
                    output,
                    family=family,
                )

    def test_stage_oom_revision_is_attempt_bundle_and_batch_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor = object.__new__(FamilySupervisor)
            supervisor.campaign_root = root
            supervisor.config = type(
                "Config",
                (),
                {"raw": {"training": {"output_subdirectory": "runs"}}},
            )()
            supervisor.precision_role_plans = {
                "praxis": {"plan_sha256": "c" * 64}
            }
            family = FamilyTopology(
                name="praxis",
                nodes=1,
                world_size=4,
                relative_node=0,
                expert_parallel_size=4,
                expert_replicas=1,
                manifest=root / "praxis.yaml",
            )
            output = root / "runs" / "praxis"
            oom_root = output / "posttraining" / "praxis" / "oom"
            oom_root.mkdir(parents=True)
            request = {
                "schema": "metis.posttraining-oom-revision-request/v1",
                "family": "praxis",
                "stage": "context_extension",
                "parent_checkpoint_sha256": "a" * 64,
                "bundle_manifest_sha256": "b" * 64,
                "precision_role_plan_sha256": "c" * 64,
                "prior_batch_migration_sha256": None,
                "rank": 2,
                "world_size": 4,
                "slurm_job_id": "4321",
                "slurm_restart_count": 0,
                "sequence_length": 163_840,
                "phase": "forward_backward",
                "resume": {
                    "epoch": 0,
                    "next_global_batch": 8,
                    "optimizer_step": 2,
                    "campaign_token_cursor": 1_000_000_000_000,
                },
                "current": {
                    "micro_batch_size": 4,
                    "gradient_accumulation": 2,
                },
                "proposed": {
                    "micro_batch_size": 2,
                    "gradient_accumulation": 4,
                },
                "revision_available": True,
                "exception_type": "OutOfMemoryError",
                "created_unix": 1,
            }
            request["request_sha256"] = json_sha256(request)
            request_path = oom_root / "context_extension-rank00002.json"
            atomic_write_json(request_path, request)
            with mock.patch.dict(
                "os.environ",
                {"SLURM_JOB_ID": "4321", "SLURM_RESTART_COUNT": "0"},
                clear=False,
            ):
                summary = supervisor.revise_posttraining_batch_after_oom(
                    family=family,
                    restart_count=0,
                )
            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(
                summary["old"]["micro_batch_size"]
                * summary["old"]["gradient_accumulation"],
                summary["new"]["micro_batch_size"]
                * summary["new"]["gradient_accumulation"],
            )
            validate_posttraining_batch_migration_for_requeue(
                summary,
                campaign_root=root,
                output_root=output,
                family=family,
            )
            request["phase"] = "tampered"
            atomic_write_json(request_path, request)
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                validate_posttraining_batch_migration_for_requeue(
                    summary,
                    campaign_root=root,
                    output_root=output,
                    family=family,
                )

    def test_requeue_checkpoint_validation_is_hash_and_size_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            config = self.config(temporary)
            family = config.families[0]
            output = temporary / "run"
            checkpoint = output / "checkpoints" / "tokens-0000000004096"
            checkpoint.mkdir(parents=True)
            artifact = checkpoint / "replicated.pt"
            artifact.write_bytes(b"checkpoint")
            manifest = {
                "schema": "metis.distributed-checkpoint/v1",
                "family": family.name,
                "world_size": family.world_size,
                "expert_parallel_size": family.expert_parallel_size,
                "expert_replica_count": family.expert_replicas,
                "global_token_cursor": 4096,
                "optimizer_step": 1,
                "autotune_profile_sha256": "a" * 64,
                "precision_role_plan_sha256": "b" * 64,
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": file_sha256(artifact),
                    }
                ],
            }
            manifest["checkpoint_sha256"] = json_sha256(manifest)
            atomic_write_json(checkpoint / "MANIFEST.json", manifest)
            atomic_write_json(
                output / "checkpoints" / "LATEST.json",
                {
                    "schema": "metis.checkpoint-latest/v1",
                    "checkpoint": checkpoint.name,
                },
            )
            state = validate_checkpoint_for_requeue(
                output,
                family=family,
                require_checkpoint=True,
            )
            self.assertEqual(state["checkpoint_sha256"], manifest["checkpoint_sha256"])
            self.assertEqual(
                state["precision_role_plan_sha256"],
                "b" * 64,
            )
            artifact.write_bytes(b"checkpoinu")
            with self.assertRaisesRegex(RuntimeError, "hash drifted"):
                validate_checkpoint_for_requeue(
                    output,
                    family=family,
                    require_checkpoint=True,
                )
            artifact.write_bytes(b"short")
            with self.assertRaisesRegex(RuntimeError, "size-drifted"):
                validate_checkpoint_for_requeue(
                    output,
                    family=family,
                    require_checkpoint=True,
                )

    def test_failure_classification_ignores_stale_oom_from_prior_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor = object.__new__(FamilySupervisor)
            supervisor.campaign_root = root
            supervisor.config = type(
                "Config",
                (),
                {
                    "raw": {
                        "autonomy": {
                            "recognized_transient_patterns": [
                                "connection reset by peer"
                            ]
                        }
                    }
                },
            )()
            previous = (
                root
                / "logs"
                / "training-attempts"
                / "restart-000"
                / "praxis.log"
            )
            previous.parent.mkdir(parents=True)
            previous.write_text("HIP out of memory\n", encoding="utf-8")
            current = (
                root
                / "logs"
                / "training-attempts"
                / "restart-001"
                / "praxis.log"
            )
            current.parent.mkdir(parents=True)
            current.write_text(
                "RuntimeError: checkpoint lineage mismatch\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                "os.environ",
                {"SLURM_RESTART_COUNT": "1"},
                clear=False,
            ):
                classification, failed = supervisor.classify_failure(
                    {"praxis": 1, "logos": 0}
                )
            self.assertEqual(classification, "trainer_failure")
            self.assertEqual(failed, ["praxis"])

    def test_missing_posttraining_index_fails_before_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            report = inspect_posttraining_release_index(config)
            self.assertFalse(report["ok"])
            self.assertIn("tokenizer_manifest", report["missing"]["praxis"])

    def test_posttraining_index_binds_every_family_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            lustre = temporary / "lustre" / "vollmerc" / "metis-1.6"
            index_path = (
                lustre
                / "releases"
                / "metis-1.6-posttraining-r1"
                / "RELEASE_INDEX.json"
            ).resolve()
            config = self.config(
                temporary,
                posttraining_index=index_path,
            )
            _base_release(lustre)
            index_path.parent.mkdir(parents=True, exist_ok=True)

            def sealed(
                path: Path,
                *,
                schema: str,
                metadata: dict,
                tokenizer_sha256: str | None,
            ) -> tuple[dict, str]:
                path.parent.mkdir(parents=True, exist_ok=True)
                payload_path = path.with_suffix(".bin")
                payload_path.write_bytes(path.name.encode("utf-8"))
                manifest = {
                    "envelope_schema": "metis.sealed-artifact/v1",
                    "schema": schema,
                    "complete": True,
                    "metadata": metadata,
                    "files": [
                        {
                            "path": payload_path.name,
                            "bytes": payload_path.stat().st_size,
                            "sha256": file_sha256(payload_path),
                        }
                    ],
                }
                if tokenizer_sha256 is not None:
                    manifest["tokenizer_sha256"] = tokenizer_sha256
                manifest["manifest_sha256"] = json_sha256(manifest)
                atomic_write_json(path, manifest)
                return manifest, manifest["manifest_sha256"]

            contract = yaml.safe_load(
                config.posttraining_contract.read_text(encoding="utf-8")
            )
            families: dict[str, dict] = {}
            expected_tokenizer_sha: str | None = None
            praxis_family_index_path: Path | None = None
            for family in config.families:
                family_root = (index_path.parent / family.name).resolve()
                family_index_path = family_root / "POSTTRAINING_RELEASE.json"
                tokenizer_path = family_root / "tokenizer" / "MANIFEST.json"
                (
                    tokenizer_record,
                    tokenizer_path,
                    tokenizer_payload,
                ) = _sealed_base_tokenizer(
                    config=config,
                    release_root=family_root,
                )
                tokenizer_sha = tokenizer_payload["manifest_sha256"]
                if expected_tokenizer_sha is None:
                    expected_tokenizer_sha = tokenizer_sha
                self.assertEqual(tokenizer_sha, expected_tokenizer_sha)
                nested: dict[str, dict] = {}
                for stage in contract["stages"]:
                    if stage.get("enabled") is not True:
                        continue
                    stage_rows: dict[str, dict] = {}
                    for requirement in stage.get("requirements", []):
                        if requirement.get("checkpoint_bound") is True:
                            generated_root = (
                                f"generated/{stage['id']}/"
                                f"{requirement['name']}"
                            )
                            hook_path = (
                                family_root
                                / "hooks"
                                / (
                                    f"generate-{stage['id']}-"
                                    f"{requirement['name']}.py"
                                )
                            )
                            hook_path.parent.mkdir(parents=True, exist_ok=True)
                            hook_path.write_text(
                                """#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

output = Path(os.environ["METIS_OUTPUT_MANIFEST"])
receipt_path = Path(os.environ["METIS_GENERATION_RECEIPT"])
output.parent.mkdir(parents=True, exist_ok=True)
receipt_path.parent.mkdir(parents=True, exist_ok=True)
output.write_text('{"generated":true}\\n', encoding="utf-8")
receipt = {
    "schema": "metis.generation-hook-receipt/v1",
    "family": os.environ["METIS_FAMILY"],
    "stage": os.environ["METIS_STAGE_ID"],
    "requirement": os.environ["METIS_REQUIREMENT_NAME"],
    "parent_checkpoint_sha256": os.environ[
        "METIS_PARENT_CHECKPOINT_SHA256"
    ],
    "executable_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "output_manifest_sha256": hashlib.sha256(
        output.read_bytes()
    ).hexdigest(),
    "success": True,
}
receipt["receipt_sha256"] = digest(receipt)
receipt_path.write_text(
    json.dumps(receipt, sort_keys=True) + "\\n",
    encoding="utf-8",
)
""",
                                encoding="utf-8",
                            )
                            hook_path.chmod(0o750)
                            stage_rows[requirement["name"]] = {
                                "schema": requirement["schema"],
                                "state": "deferred",
                                "manifest": f"{generated_root}/MANIFEST.json",
                                "generation_hook": {
                                    "executable": str(
                                        hook_path.relative_to(family_root)
                                    ),
                                    "executable_sha256": file_sha256(hook_path),
                                    "args": [],
                                    "timeout_seconds": 60,
                                    "execution": {
                                        "protocol": "rank0_only_v1",
                                        "nodes": 1,
                                        "tasks": 1,
                                        "gpus_per_task": 0,
                                    },
                                    "receipt": f"{generated_root}/RECEIPT.json",
                                    "rank_receipts": (
                                        f"{generated_root}/rank-receipts"
                                    ),
                                },
                            }
                            continue
                        metadata = dict(requirement.get("required_metadata", {}))
                        if requirement.get("family_bound") is True:
                            metadata["family"] = family.name
                        if requirement.get("generated_from_stage") is not None:
                            metadata["generated_from_stage"] = requirement[
                                "generated_from_stage"
                            ]
                        if requirement.get("checkpoint_bound") is True:
                            metadata["generated_from_checkpoint_sha256"] = "c" * 64
                        if "minimum_records" in requirement:
                            metadata["records"] = int(requirement["minimum_records"])
                        if "minimum_tokens" in requirement:
                            metadata["tokens"] = int(requirement["minimum_tokens"])
                        if "minimum_source_instructions" in requirement:
                            metadata["source_instruction_count"] = int(
                                requirement["minimum_source_instructions"]
                            )
                        artifact_path = (
                            family_root
                            / "artifacts"
                            / stage["id"]
                            / requirement["name"]
                            / "MANIFEST.json"
                        )
                        manifest, manifest_sha = sealed(
                            artifact_path,
                            schema=requirement["schema"],
                            metadata=metadata,
                            tokenizer_sha256=(
                                tokenizer_sha
                                if requirement.get("tokenizer_bound", True)
                                else None
                            ),
                        )
                        del manifest
                        stage_rows[requirement["name"]] = {
                            "schema": requirement["schema"],
                            "state": "sealed",
                            "manifest": str(artifact_path.relative_to(family_root)),
                            "sha256": file_sha256(artifact_path),
                            "manifest_sha256": manifest_sha,
                        }
                    nested[stage["id"]] = stage_rows
                family_index = {
                    "schema": "metis.posttraining-release-index/v1",
                    "family": family.name,
                    "pipeline_sha256": file_sha256(config.posttraining_contract),
                    "tokenizer_manifest": tokenizer_record,
                    "requirements": nested,
                }
                family_index["index_sha256"] = json_sha256(family_index)
                atomic_write_json(family_index_path, family_index)
                if family.name == "praxis":
                    praxis_family_index_path = family_index_path
                families[family.name] = {
                    "path": str(family_index_path.relative_to(index_path.parent)),
                    "sha256": file_sha256(family_index_path),
                    "index_sha256": family_index["index_sha256"],
                }
            inventory = []
            for payload_path in sorted(
                path
                for path in index_path.parent.rglob("*")
                if path.is_file()
            ):
                inventory.append(
                    {
                        "path": str(payload_path.relative_to(index_path.parent)),
                        "bytes": payload_path.stat().st_size,
                        "sha256": file_sha256(payload_path),
                    }
                )
            deep = {
                "schema": "metis.posttraining-release-deep-verification/v1",
                "posttraining_contract_sha256": file_sha256(
                    config.posttraining_contract
                ),
                "files": inventory,
                "file_count": len(inventory),
                "total_bytes": sum(row["bytes"] for row in inventory),
                "complete": True,
            }
            deep["receipt_sha256"] = json_sha256(deep)
            deep_path = index_path.parent / "DEEP_VERIFICATION.json"
            atomic_write_json(deep_path, deep)
            umbrella = {
                "schema": "metis.posttraining-release-umbrella/v1",
                "posttraining_contract_sha256": file_sha256(
                    config.posttraining_contract
                ),
                "families": families,
                "deep_verification": {
                    "path": deep_path.name,
                    "sha256": file_sha256(deep_path),
                    "receipt_sha256": deep["receipt_sha256"],
                },
            }
            umbrella["umbrella_sha256"] = json_sha256(umbrella)
            atomic_write_json(index_path, umbrella)
            report = inspect_posttraining_release_index(config)
            self.assertTrue(report["ok"], report.get("errors"))
            praxis = environment_for_family(
                report,
                "praxis",
                config=config,
            )
            self.assertEqual(
                praxis,
                {"METIS_POSTTRAINING_RELEASE_INDEX": str(index_path.resolve())},
            )
            topology = ParallelTopology(
                family="praxis",
                world_size=1,
                rank=0,
                local_rank=0,
                expert_parallel_size=1,
                expert_replica_count=1,
                expert_group=None,
                expert_group_ranks=(0,),
                expert_data_group=None,
                expert_data_group_ranks=(0,),
                dense_data_group=None,
            )
            deferred_environment = {
                **praxis,
                "METIS_POSTTRAINING_DEEP_VERIFICATION": str(deep_path),
                "METIS_POSTTRAINING_DEEP_VERIFICATION_FILE_SHA256": (
                    file_sha256(deep_path)
                ),
                "METIS_POSTTRAINING_DEEP_VERIFICATION_RECEIPT_SHA256": (
                    deep["receipt_sha256"]
                ),
                "SLURM_JOB_ID": "1234",
                "SLURM_RESTART_COUNT": "0",
            }
            with mock.patch.dict(
                "os.environ", deferred_environment, clear=False
            ):
                loaded = _load_release_index(
                    family="praxis",
                    pipeline_sha256=file_sha256(config.posttraining_contract),
                    topology=topology,
                )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            assert praxis_family_index_path is not None
            self.assertEqual(loaded[1], praxis_family_index_path.resolve())
            deferred = loaded[0]["requirements"]["deepseek_dpd_pilot"][
                "deepseek_dpd_pilot_data"
            ]
            output_root = temporary / "trainer-output" / "posttraining" / "praxis"
            with mock.patch.dict(
                "os.environ", deferred_environment, clear=False
            ), self.assertRaises(DeferredMaterialization) as raised:
                _materialize_generation_hook(
                    release_index=loaded,
                    record=deferred,
                    family="praxis",
                    stage_id="deepseek_dpd_pilot",
                    requirement_name="deepseek_dpd_pilot_data",
                    parent_checkpoint_sha256="f" * 64,
                    topology=topology,
                    output_root=output_root,
                    stage_bindings={
                        "parent_policy_checkpoint": {
                            "stage_id": "overall_sft",
                            "checkpoint_path": "/sealed/parent",
                            "checkpoint_sha256": "f" * 64,
                            "checkpoint_receipt": "/sealed/parent-receipt",
                            "checkpoint_contract": {},
                        },
                        "dpd_reference_checkpoint": {
                            "stage_id": "overall_sft",
                            "checkpoint_path": "/sealed/parent",
                            "checkpoint_sha256": "f" * 64,
                            "checkpoint_receipt": "/sealed/parent-receipt",
                            "checkpoint_contract": {},
                        },
                    },
                )
            request_path = raised.exception.request_path
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(
                request["request_sha256"],
                json_sha256(request, omit=("request_sha256",)),
            )
            self.assertFalse(
                Path(request["hook"]["output_manifest"]).exists(),
                "the trainer must not run the deferred subprocess",
            )

    def test_cxi_snapshot_and_mfu_are_evidence_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run" / "cxi" / "cxi0"
            root.mkdir(parents=True)
            (root / "sct_timeouts").write_text("0\n", encoding="utf-8")
            row = snapshot_cxi([root.parent])
            self.assertTrue(row["ok"])
            self.assertEqual(
                mfu(
                    estimated_train_flops=50.0,
                    elapsed_seconds=1.0,
                    world_size=1,
                    dense_peak_flops_per_apu=100.0,
                ),
                0.5,
            )
            with self.assertRaisesRegex(ValueError, "Implausible MFU"):
                mfu(
                    estimated_train_flops=200.0,
                    elapsed_seconds=1.0,
                    world_size=1,
                    dense_peak_flops_per_apu=100.0,
                )


if __name__ == "__main__":
    unittest.main()
