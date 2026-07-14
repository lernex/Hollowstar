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


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight Metis-1.5 JAX TPU v6e-8 readiness.")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "configs/metis15_manifest.json")
    parser.add_argument("--stage", choices=["pretrain", "continued_pretrain"], default="pretrain")
    parser.add_argument("--local-batch-size", type=int, default=None)
    parser.add_argument("--skip-device-check", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    print("Metis-1.5 JAX TPU v6e preflight")
    print(f"  python: {platform.python_version()} ({sys.executable})")
    print(f"  manifest: {args.manifest}")
    if sys.version_info < (3, 11):
        failures.append("Python 3.11+ is required for the JAX TPU lane.")

    for module_name in ("jax", "jax.numpy"):
        try:
            module = importlib.import_module(module_name)
            if module_name == "jax":
                print(f"  jax: {getattr(module, '__version__', 'unknown')}")
        except Exception as exc:
            failures.append(f"{module_name} import failed: {exc}")

    try:
        from metis_mamba.jax_metis import load_manifest_config

        cfg, train_cfg = load_manifest_config(args.manifest, stage=args.stage)
        local_batch_size = args.local_batch_size or train_cfg.local_batch_size
        cfg.validate(local_batch_size=local_batch_size)
        print(
            "  model: "
            f"layers={cfg.n_layer} d_model={cfg.d_model} block={cfg.block_size} "
            f"experts={cfg.moe_num_experts} top_k={cfg.moe_top_k} latent={cfg.moe_routed_latent_size}"
        )
        print(
            "  v6e layout: "
            f"world=8 experts_per_rank={cfg.experts_per_rank} capacity={cfg.capacity_for_batch(local_batch_size)} "
            f"stage={args.stage}"
        )
        if args.stage == "pretrain" and cfg.mor_enabled:
            failures.append("pretrain stage must not enable MoR.")
        if args.stage == "continued_pretrain" and not cfg.mor_enabled:
            failures.append("continued_pretrain stage must enable dynamic token MoR.")
        if args.stage == "continued_pretrain" and cfg.mor_compute_mode != "static_packed_hard":
            failures.append("continued_pretrain must use static_packed_hard MoR compute.")
        if args.stage == "continued_pretrain" and abs(cfg.mor_router_temperature - 1.0) > 1e-9:
            failures.append("continued_pretrain MoR router_temperature must load as 1.0 for the current JAX contract.")
    except Exception as exc:
        failures.append(f"manifest/JAX contract validation failed: {exc}")

    if not args.skip_device_check:
        try:
            import jax

            devices = jax.devices()
            print(f"  devices: {devices}")
            tpu_devices = [device for device in devices if str(device.platform).lower() == "tpu"]
            if len(tpu_devices) != 8:
                failures.append(f"expected 8 TPU devices for v6e-8; saw {len(tpu_devices)} TPU devices.")
        except Exception as exc:
            failures.append(f"jax device enumeration failed: {exc}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    model = manifest.get("model", {})
    hardware = manifest.get("hardware", {})
    if hardware.get("runtime") != "JAX/libTPU":
        failures.append("manifest hardware.runtime must be JAX/libTPU.")
    if hardware.get("target_cluster") != "google_cloud_tpu_v6e_8_chip_jax_static_expert_parallel":
        failures.append("manifest target_cluster must be the JAX v6e-8 expert-parallel lane.")
    if hardware.get("requirements_file") != "requirements-jax-tpu-train.txt":
        failures.append("manifest must point at requirements-jax-tpu-train.txt.")
    if hardware.get("local_readiness_script") != "scripts/metis15_jax_tpu_v6e_local_readiness.sh":
        failures.append("manifest local_readiness_script must point at the JAX TPU local readiness runner.")
    if hardware.get("sharding_report_script") != "scripts/metis15_jax_tpu_v6e_sharding_report.py":
        failures.append("manifest sharding_report_script must point at the JAX TPU sharding report.")
    if hardware.get("compile_probe_script") != "scripts/metis15_jax_tpu_v6e_compile_probe.sh":
        failures.append("manifest compile_probe_script must point at the JAX TPU compile probe.")
    if hardware.get("perf_sweep_script") != "scripts/metis15_jax_tpu_v6e_perf_sweep.sh":
        failures.append("manifest perf_sweep_script must point at the JAX TPU sweep.")
    if hardware.get("quality_canary_script") != "scripts/metis15_jax_tpu_v6e_quality_canary.sh":
        failures.append("manifest quality_canary_script must point at the JAX TPU canary.")
    if hardware.get("log_analyzer_script") != "scripts/analyze_metis15_jax_tpu_logs.py":
        failures.append("manifest log_analyzer_script must point at the JAX log analyzer.")
    if hardware.get("gcs_checkpoint_sync") and hardware.get("gcs_checkpoint_env") != "METIS15_JAX_GCS_CHECKPOINT_DIR":
        failures.append("manifest gcs_checkpoint_env must be METIS15_JAX_GCS_CHECKPOINT_DIR.")
    if hardware.get("gcs_data_env") != "METIS15_JAX_DATA_GCS_URI":
        failures.append("manifest gcs_data_env must be METIS15_JAX_DATA_GCS_URI.")
    if not hardware.get("remat_layers", False):
        failures.append("hardware.remat_layers must stay enabled for the full v6e JAX lane.")
    throughput = hardware.get("throughput", {})
    if not throughput.get("compile_probe_requires_jax_log_compiles", False):
        failures.append("throughput.compile_probe_requires_jax_log_compiles must be true.")
    if throughput.get("safe_promotion_gate") != (
        "scripts/summarize_metis15_jax_tpu_sweep.py plus fixed-batch JAX quality canary must pass before promoting a tok/s setting"
    ):
        failures.append("throughput.safe_promotion_gate must reference the JAX sweep and JAX quality canary.")
    expected_sweep_dims = {
        "local_batch_size",
        "grad_accum_steps",
        "block_size",
        "expert_capacity_factor",
        "remat_mode",
        "dtype",
        "expert_execution",
    }
    if set(throughput.get("sweep_dimensions", [])) != expected_sweep_dims:
        failures.append("throughput.sweep_dimensions must list the JAX v6e tuning knobs.")
    if model.get("moe_backend") != "jax_static_sort_pack":
        failures.append("model.moe_backend must be jax_static_sort_pack.")
    if model.get("moe_dispatch_mode") != "static_sort_pack":
        failures.append("model.moe_dispatch_mode must be static_sort_pack.")
    if not model.get("moe_graphable"):
        failures.append("model.moe_graphable must be true for the JAX lane.")
    optimizer = manifest.get("optimizer", {})
    if optimizer.get("name") != "muon_adamw":
        failures.append("manifest optimizer.name must be muon_adamw.")
    if optimizer.get("include_routed_experts"):
        failures.append("routed experts must stay on AdamW by default.")
    if optimizer.get("muon_scale_mode") != "match_rms_adamw":
        failures.append("Muon scale mode must stay match_rms_adamw.")
    gcs_checkpoint_dir = os.environ.get("METIS15_JAX_GCS_CHECKPOINT_DIR", "")
    if gcs_checkpoint_dir:
        if not gcs_checkpoint_dir.startswith("gs://"):
            failures.append("METIS15_JAX_GCS_CHECKPOINT_DIR must start with gs:// when set.")
        if shutil.which("gcloud") is None:
            failures.append("METIS15_JAX_GCS_CHECKPOINT_DIR requires the gcloud CLI on PATH.")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("metis15_jax_tpu_preflight_ok")


if __name__ == "__main__":
    main()
