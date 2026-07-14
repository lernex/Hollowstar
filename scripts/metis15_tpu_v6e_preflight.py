#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts"
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"FAIL: manifest not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FAIL: manifest is not valid JSON: {path}: {exc}") from exc


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"FAIL: manifest value {name} must be an integer; got {value!r}") from None


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"FAIL: value {name} must be numeric; got {value!r}") from None


def _check_import(name: str, failures: list[str], *, required: bool = True) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - exact import error is environment-specific.
        message = f"{name} import failed: {exc}"
        if required:
            failures.append(message)
        else:
            print(f"WARN: {message}")
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight Google Cloud TPU v6e-8 readiness for Metis-1.5.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-stage", choices=["pretrain", "continued_pretrain"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--data-gcs-uri", default="")
    parser.add_argument("--checkpoint-gcs-uri", default="")
    parser.add_argument("--dispatch-pack-impl", default="index_add")
    parser.add_argument("--router-override", default="learned")
    parser.add_argument("--attention-kernel", default="sdpa")
    parser.add_argument("--optimizer", default="muon_adamw")
    parser.add_argument("--hybrid-adamw-impl", default="loop")
    parser.add_argument("--muon-scale-mode", default="match_rms_adamw")
    parser.add_argument("--ce-logits-dtype", default="float32")
    parser.add_argument("--qk-clip-threshold", type=float, default=100.0)
    parser.add_argument("--expert-capacity-factor", type=float, default=4.0)
    parser.add_argument(
        "--skip-device-check",
        action="store_true",
        help="Skip torch_xla TPU device enumeration. Intended only for local dry-runs.",
    )
    parser.add_argument(
        "--allow-world-size-mismatch",
        action="store_true",
        help="Allow non-8 world sizes for local experiments; production v6e-8 should not set this.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    failures: list[str] = []
    warnings: list[str] = []

    manifest = _load_manifest(args.manifest)
    model = manifest.get("model", {})
    optimizer = manifest.get("optimizer", {})
    hardware = manifest.get("hardware", {})
    stability = hardware.get("stability", {}) if isinstance(hardware, dict) else {}

    print("Metis-1.5 TPU v6e preflight")
    print(f"  python: {platform.python_version()} ({sys.executable})")
    print(f"  manifest: {args.manifest}")
    print(f"  stage: {args.train_stage}")
    print(f"  target world size: {args.world_size}")

    if sys.version_info < (3, 10):
        failures.append(f"Python 3.10+ is required for the TPU lane; got {platform.python_version()}")

    torch = _check_import("torch", failures)
    if torch is not None:
        print(f"  torch: {getattr(torch, '__version__', 'unknown')}")

    trainer = _check_import("train_metis15_tpu", failures)
    if trainer is not None:
        kernels = set(getattr(trainer, "ATTENTION_KERNEL_CHOICES", ()))
        if "nki" in kernels:
            failures.append("TPU trainer exposes NKI attention kernels; this path must stay Google/PJRT-only.")
        if args.attention_kernel not in kernels:
            failures.append(f"attention kernel {args.attention_kernel!r} is not supported by the TPU trainer.")
        print(f"  trainer attention kernels: {','.join(sorted(kernels))}")

    torchrun_bin = os.environ.get("TORCHRUN_BIN", "torchrun")
    if shutil.which(torchrun_bin) is None:
        try:
            importlib.import_module("torch.distributed.run")
            print("  torchrun: using python -m torch.distributed.run fallback")
        except Exception as exc:
            failures.append(
                f"{torchrun_bin} was not found on PATH and torch.distributed.run is unavailable: {exc}"
            )
    else:
        print(f"  torchrun: {shutil.which(torchrun_bin)}")

    if not args.allow_world_size_mismatch and int(args.world_size) != 8:
        failures.append(f"v6e-8 production launch expects world size 8; got {args.world_size}.")

    pjrt_device = os.environ.get("PJRT_DEVICE", "")
    if not args.skip_device_check and pjrt_device.upper() != "TPU":
        failures.append(f"PJRT_DEVICE must be TPU for a TPU launch; got {pjrt_device!r}.")

    if not args.skip_device_check:
        torch_xla = _check_import("torch_xla", failures)
        xm = _check_import("torch_xla.core.xla_model", failures)
        if torch_xla is not None:
            print(f"  torch_xla: {getattr(torch_xla, '__version__', 'unknown')}")
        if xm is not None:
            try:
                devices = list(xm.get_xla_supported_devices("TPU"))
                print(f"  TPU devices: {devices}")
                if len(devices) != int(args.world_size):
                    failures.append(
                        f"torch_xla sees {len(devices)} TPU devices, expected {args.world_size}: {devices}"
                    )
            except Exception as exc:
                failures.append(f"torch_xla could not enumerate TPU devices: {exc}")

    gcs_needed = bool(args.data_gcs_uri or args.checkpoint_gcs_uri)
    if gcs_needed and shutil.which("gcloud") is None and shutil.which("gsutil") is None:
        failures.append("GCS sync requested but neither gcloud nor gsutil is on PATH.")

    if not args.synthetic_data:
        meta_path = args.data_dir / "meta.json"
        train_path = args.data_dir / "train.bin"
        if not (meta_path.is_file() and train_path.is_file()) and not args.data_gcs_uri:
            failures.append(
                f"training data missing at {meta_path} / {train_path}, and no --data-gcs-uri was provided."
            )
    else:
        warnings.append("synthetic data enabled; this checks compile/perf plumbing, not real data learning.")

    if not args.manifest.is_file():
        failures.append(f"manifest path is missing: {args.manifest}")
    if not (ROOT_DIR / "scripts" / "train_metis15_tpu.py").is_file():
        failures.append("scripts/train_metis15_tpu.py is missing.")

    num_experts = _as_int(model.get("moe_num_experts"), "model.moe_num_experts")
    top_k = _as_int(model.get("moe_top_k"), "model.moe_top_k")
    n_layer = _as_int(model.get("n_layer"), "model.n_layer")
    block_size = int(args.block_size or _as_int(model.get("block_size"), "model.block_size"))
    if num_experts % int(args.world_size) != 0:
        failures.append(f"moe_num_experts={num_experts} must divide world_size={args.world_size}.")
    if top_k <= 0 or top_k > num_experts:
        failures.append(f"moe_top_k={top_k} is invalid for moe_num_experts={num_experts}.")
    if args.dispatch_pack_impl in {"group_static", "group_static_gather"} and int(args.world_size) % top_k != 0:
        failures.append(
            f"{args.dispatch_pack_impl} requires world_size divisible by top_k "
            f"(world_size={args.world_size}, top_k={top_k})."
        )
    if args.dispatch_pack_impl in {"balanced_static", "group_static", "group_static_gather", "dense_all_experts"}:
        if num_experts != int(args.world_size):
            failures.append(
                f"{args.dispatch_pack_impl} requires one expert per rank "
                f"(moe_num_experts={num_experts}, world_size={args.world_size})."
            )
    if args.dispatch_pack_impl == "balanced_static" and args.router_override != "force_balanced":
        failures.append("balanced_static dispatch requires router_override=force_balanced.")

    expected_valid = int(args.world_size) * int(args.local_batch_size) * block_size * top_k * n_layer
    tokens_per_step = int(args.world_size) * int(args.local_batch_size) * int(args.grad_accum_steps) * block_size
    print(f"  tokens/optimizer-step: {tokens_per_step:,}")
    print(f"  expected valid assignments/logged step: {expected_valid:,}")
    print(f"  experts/rank: {num_experts // max(1, int(args.world_size))}")

    normalized_optimizer = args.optimizer.replace("-", "_")
    if normalized_optimizer not in {"muon_adamw", "hybrid_muon_adamw"}:
        failures.append(f"optimizer must be AdamW/Muon hybrid for Metis-1.5 TPU; got {args.optimizer!r}.")
    if str(args.hybrid_adamw_impl) != "loop":
        failures.append("TPU launch should use hybrid_adamw_impl=loop to avoid foreach/XLA instability.")
    if str(args.muon_scale_mode) != "match_rms_adamw":
        failures.append("Muon scale mode should be match_rms_adamw for the Kimi/Moonshot-style recipe.")
    if bool(optimizer.get("include_routed_experts", False)) or _bool_env("METIS15_TPU_MUON_INCLUDE_ROUTED_EXPERTS", False):
        failures.append("routed experts must stay on AdamW unless running a deliberate ablation.")
    if not bool(optimizer.get("master_weights", False)) and not _bool_env("METIS15_TPU_OPTIMIZER_MASTER_WEIGHTS", True):
        failures.append("FP32 optimizer master weights should be enabled for the BF16 TPU lane.")

    if str(args.ce_logits_dtype) != "float32":
        failures.append("ce_logits_dtype must stay float32 until a TPU quality ablation proves otherwise.")
    qk_threshold = _as_float(args.qk_clip_threshold, "qk_clip_threshold")
    if qk_threshold <= 0:
        failures.append("QK clipping is disabled; keep MuonClip/QK-clip on for production TPU training.")
    manifest_qk = stability.get("qk_clip", {}) if isinstance(stability, dict) else {}
    if isinstance(manifest_qk, dict) and not bool(manifest_qk.get("enabled", False)):
        failures.append("manifest hardware.stability.qk_clip.enabled must be true.")
    if _as_float(args.expert_capacity_factor, "expert_capacity_factor") < 1.0:
        failures.append("expert_capacity_factor below 1.0 risks token drops before the router is trained.")

    blocked_env = sorted(name for name in os.environ if name.startswith(("NEURON_", "FI_EFA_", "AWS_NEURON_")))
    if blocked_env:
        warnings.append(f"Neuron/Trainium environment variables are present and ignored by TPU path: {blocked_env}")

    if warnings:
        for warning in warnings:
            print(f"WARN: {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("metis15_tpu_preflight_ok")


if __name__ == "__main__":
    main()
