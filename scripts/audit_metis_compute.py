from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_config_class():
    config_path = Path(__file__).resolve().parents[1] / "src" / "metis_mamba" / "config.py"
    spec = importlib.util.spec_from_file_location("metis_mamba_config_standalone", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config module from {config_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MetisMambaConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Print dense and MoE parameter-application accounting.")
    parser.add_argument("--manifest", default="configs/metis15_manifest.json")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    MetisMambaConfig = _load_config_class()
    config = MetisMambaConfig.from_dict(manifest["model"])
    config.validate()

    audit = config.param_application_audit()

    print(f"name: {manifest.get('name', config.name)}")
    print(f"config_estimate_params: {config.estimate_params():,}")
    print(f"config_estimate_active_params_depth1: {config.estimate_active_params(1.0):,}")
    print(f"ffn_type: {audit['ffn_type']}")
    print(f"d_model: {audit['d_model']:,}")
    print(f"embedding_params: {audit['embedding_params']:,}")
    print(f"attention_apps_per_layer: {audit['attention_apps_per_layer']:,}")
    if not config.uses_moe:
        print(f"dense_mlp_param_apps_per_layer: {audit['dense_mlp_param_apps_per_layer']:,}")
        print(f"rough_total_param_apps_per_token: {audit['rough_total_param_apps_per_token']:,}")
        print(f"estimated_train_flops_per_token: {audit['estimated_train_flops_per_token']:,}")
        return

    print(f"latent_dim: {audit['latent_dim']:,}")
    print(f"num_experts: {audit['num_experts']:,}")
    print(f"top_k: {audit['top_k']:,}")
    print(f"routing_units_per_token: {audit['routing_units_per_token']:,}")
    print(f"expert_hidden: {audit['expert_hidden']:,}")
    print(f"moe_activation: {audit['moe_activation']}")
    print(f"expert_param_apps_per_assignment: {audit['expert_param_apps_per_assignment']:,}")
    print(f"routed_expert_param_apps_per_layer: {audit['routed_expert_param_apps_per_layer']:,}")
    print(f"shared_expert_param_apps_per_layer: {audit['shared_expert_param_apps_per_layer']:,}")
    print(f"latent_projection_apps_per_layer: {audit['latent_projection_apps_per_layer']:,}")
    print(f"router_projection_and_match_apps_per_layer: {audit['router_projection_and_match_apps_per_layer']:,}")
    print(f"rough_total_param_apps_per_token: {audit['rough_total_param_apps_per_token']:,}")
    print(f"estimated_train_flops_per_token: {audit['estimated_train_flops_per_token']:,}")


if __name__ == "__main__":
    main()
