#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from metis_mamba import MetisMambaConfig
from metis_mamba.optim import build_optimizer_from_args
from analyze_metis15_tpu_logs import audit, parse_log
from train_metis15_tpu import Runtime, XlaMetisForCausalLM, make_profile_totals


def tiny_tpu_config() -> MetisMambaConfig:
    return MetisMambaConfig(
        name="Metis-1.5-tpu-contract-smoke",
        vocab_size=64,
        block_size=16,
        d_model=32,
        n_layer=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        tie_embeddings=True,
        attention_backend="eager",
        training_mode="static_dense_pretrain",
        mor_enabled=False,
        mor_runtime_mode="disabled",
        ffn_type="single_latent_moe",
        moe_num_experts=4,
        moe_top_k=2,
        moe_shared_experts=1,
        moe_num_heads=1,
        moe_expert_intermediate_size=32,
        moe_router_latent_size=16,
        moe_routed_latent_size=16,
        moe_activation="squared_relu",
        moe_router_score="sigmoid",
        moe_single_latent_router_input="hidden",
        moe_balance_strategy="aux_loss_free_bias",
        moe_balance_bias_update_rate=1e-3,
        moe_aux_loss_coef=1e-4,
        moe_backend="torch_bmm",
        moe_dispatch_mode="bucketed",
        moe_expert_parallel_size=1,
        low_precision_mode="none",
        lm_loss_impl="standard",
    )


def tiny_runtime(device: torch.device) -> Runtime:
    return Runtime(
        device=device,
        device_kind=device.type,
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_xla=False,
        xm=None,
    )


def build_tiny_model(device: torch.device, *, qk_clip_enabled: bool = True) -> XlaMetisForCausalLM:
    config = tiny_tpu_config()
    runtime = tiny_runtime(device)
    model = XlaMetisForCausalLM(
        config,
        runtime=runtime,
        capacity_factor=4.0,
        capacity=None,
        dispatch_pack_impl="index_add",
        router_override="learned",
        attention_mode="real",
        attention_kernel="eager_gqa",
        qk_clip_enabled=qk_clip_enabled,
        moe_mode="real",
        loss_mode="real_ce",
        ce_logits_dtype="float32",
        ce_impl="cross_entropy",
        activation_checkpointing="none",
        activation_checkpoint_layer_interval=1,
        track_route_metrics=True,
        balanced_static_layout="indexed",
        balanced_static_router_weights="uniform",
        balanced_static_router_input="hidden",
        expert_activation_safety="clamp",
    ).to(device=device, dtype=torch.float32)
    return model


def optimizer_args() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="muon_adamw",
        lr=3e-3,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.1,
        fused_adamw=False,
        xla_stable_adamw=False,
        hybrid_adamw_impl="loop",
        optimizer_master_weights=True,
        muon_beta=0.95,
        muon_ns_steps=2,
        muon_lr_scale=1.0,
        muon_scale_mode="match_rms_adamw",
        muon_include_routed_experts=False,
    )


def assert_optimizer_group_contract(device: torch.device, manifest: dict[str, object]) -> None:
    model = build_tiny_model(device)
    optimizer, summary = build_optimizer_from_args(model, optimizer_args(), manifest.get("optimizer"))
    if summary is None or summary.name != "muon_adamw":
        raise AssertionError("Expected AdamW/Muon hybrid optimizer summary.")
    if summary.muon_params <= 0 or summary.adamw_params <= 0:
        raise AssertionError("Expected both Muon and AdamW parameter groups.")
    if summary.routed_experts_muon:
        raise AssertionError("Routed experts must stay on AdamW by default.")
    name_by_id = {id(param): name for name, param in model.named_parameters()}
    mode_by_name: dict[str, str] = {}
    for group in optimizer.param_groups:
        mode = str(group.get("optimizer", "adamw"))
        for param in group["params"]:
            mode_by_name[name_by_id[id(param)]] = mode
    expected = {
        "layers.0.attn.qkv_proj.weight": "muon",
        "layers.0.attn.o_proj.weight": "muon",
        "layers.0.moe.down_proj.weight": "muon",
        "layers.0.moe.up_proj.weight": "muon",
        "layers.0.moe.router.weight": "adamw",
        "layers.0.moe.experts.0.up.weight": "adamw",
        "embed_tokens.weight": "adamw",
    }
    for name, expected_mode in expected.items():
        actual_mode = mode_by_name.get(name)
        if actual_mode != expected_mode:
            raise AssertionError(f"Optimizer group mismatch for {name}: {actual_mode} vs {expected_mode}.")
    print(
        "tpu_optimizer_grouping_ok "
        f"muon_params={summary.muon_params} adamw_params={summary.adamw_params}",
        flush=True,
    )


@torch.no_grad()
def assert_qk_clip_contract(device: torch.device) -> None:
    model = build_tiny_model(device)
    runtime = tiny_runtime(device)
    optimizer, _summary = build_optimizer_from_args(model, optimizer_args(), {"name": "muon_adamw", "master_weights": True})
    attn = model.layers[0].attn
    optimizer.state[attn.qkv_proj.weight]["master_param"] = attn.qkv_proj.weight.detach().float().clone()
    attn.qk_clip_max_logits = torch.full_like(attn.qk_clip_max_logits, 400.0)
    q_before = attn.qkv_proj.weight[: attn.q_dim].detach().norm()
    k_before = attn.qkv_proj.weight[attn.q_dim : attn.q_dim + attn.kv_dim].detach().norm()
    stats = model.apply_qk_clip(threshold=100.0, alpha=0.5, runtime=runtime)
    optimizer.sync_master_params_from_model(model.qk_clip_parameters())
    q_after = attn.qkv_proj.weight[: attn.q_dim].detach().norm()
    k_after = attn.qkv_proj.weight[attn.q_dim : attn.q_dim + attn.kv_dim].detach().norm()
    torch.testing.assert_close(stats[0], torch.tensor(400.0, device=device), rtol=0.0, atol=1e-5)
    torch.testing.assert_close(stats[1], torch.tensor(0.5, device=device), rtol=0.0, atol=1e-5)
    if int(stats[2].item()) != attn.num_heads:
        raise AssertionError(f"Expected all heads clipped, got {float(stats[2])}.")
    if not (q_after < q_before and k_after < k_before):
        raise AssertionError("QK clip did not shrink Q and K projection weights.")
    torch.testing.assert_close(
        optimizer.state[attn.qkv_proj.weight]["master_param"],
        attn.qkv_proj.weight.detach().float(),
        rtol=0.0,
        atol=0.0,
    )
    profile = make_profile_totals()
    if "qk_clip_s" not in profile:
        raise AssertionError("Throughput profile must expose qk_clip_s.")
    print(
        "qk_clip_contract_ok "
        f"max_logit={float(stats[0]):.1f} min_scale={float(stats[1]):.3f} scaled_heads={int(stats[2].item())}",
        flush=True,
    )


def assert_tiny_moe_learning_contract(device: torch.device, manifest: dict[str, object]) -> None:
    torch.manual_seed(20260530)
    model = build_tiny_model(device)
    model.train()
    runtime = tiny_runtime(device)
    optimizer, _summary = build_optimizer_from_args(model, optimizer_args(), manifest.get("optimizer"))
    row = torch.arange(model.config.block_size, device=device, dtype=torch.long)
    input_ids = torch.stack((row, torch.remainder(row * 3 + 5, model.config.vocab_size)), dim=0)
    input_ids = torch.remainder(input_ids, model.config.vocab_size)
    losses: list[float] = []
    for _step in range(35):
        optimizer.zero_grad(set_to_none=True)
        model.reset_qk_clip_stats()
        out = model(input_ids, labels=input_ids)
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise AssertionError("Expected finite tiny MoE training loss.")
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise AssertionError(f"Nonfinite gradient in {name}.")
        optimizer.step()
        model.apply_qk_clip(threshold=100.0, alpha=0.5, runtime=runtime)
        losses.append(float(loss.detach().cpu()))
    if losses[-1] >= losses[0] * 0.97:
        raise AssertionError(f"Tiny MoE loss did not improve enough: start={losses[0]:.6f} end={losses[-1]:.6f}.")
    print(
        "tiny_moe_learning_ok "
        f"start_loss={losses[0]:.6f} end_loss={losses[-1]:.6f}",
        flush=True,
    )


def assert_log_audit_contract() -> None:
    log_text = """Launching Metis-1.5 pretrain on Google Cloud TPU v6e
  local batch size: 1
  world_size: 1
  config: layers=1 d_model=32 block=16
  moe: experts=4 local=4 top_k=2 latent=16 hidden=32
step      1 | loss 4.1602 | lm 4.1602 | moe_aux 0.00001 | valid_assign 32 | tok/s 1,060 | step_s 0.015 | lr 3.000000e-03 | tok_seen 0.0000B
profile_components step=1 data_s=0.0000 fwd_bwd_s=0.0132 grad_sync_s=0.0010 optim_s=0.0006 qk_clip_s=0.0001 mark_s=0.0000 finite_s=0.0000 log_wait_s=0.0000 p50_step_s=0.0150 p95_step_s=0.0150
qk_clip step=1 max_logit=0.047 min_scale=1.000000 scaled_heads=0
step      2 | loss 4.0268 | lm 4.0268 | moe_aux 0.00002 | valid_assign 32 | tok/s 9,057 | step_s 0.002 | lr 1.500000e-03 | tok_seen 0.0000B
profile_components step=2 data_s=0.0000 fwd_bwd_s=0.0012 grad_sync_s=0.0001 optim_s=0.0004 qk_clip_s=0.0001 mark_s=0.0000 finite_s=0.0000 log_wait_s=0.0000 p50_step_s=0.0018 p95_step_s=0.0018
qk_clip step=2 max_logit=0.041 min_scale=1.000000 scaled_heads=0
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "train.log"
        path.write_text(log_text, encoding="utf-8")
        summary = parse_log(path)
        args = SimpleNamespace(
            min_logged_steps=2,
            require_profile=True,
            require_qk_clip=True,
            require_expert_hist=False,
            require_loss_decrease=True,
            min_loss_drop_frac=0.01,
            max_final_loss=None,
            expected_valid_assign=None,
            min_valid_assign_frac=0.99,
            max_expert_drop_frac=0.01,
            max_qk_logit=1000.0,
            min_toks_per_s=None,
            perf_warmup_steps=0,
        )
        failures, warnings = audit(summary, args)
    if warnings:
        raise AssertionError(f"Unexpected log audit warnings: {warnings}")
    if failures:
        raise AssertionError(f"Unexpected log audit failures: {failures}")
    print("tpu_log_audit_contract_ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT_DIR / "configs/metis15_manifest.json"))
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    device = torch.device("cpu")
    assert_optimizer_group_contract(device, manifest)
    assert_qk_clip_contract(device)
    assert_tiny_moe_learning_contract(device, manifest)
    assert_log_audit_contract()


if __name__ == "__main__":
    main()
