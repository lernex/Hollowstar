from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from metis_mamba import MetisMambaConfig, build_model


def tiny_config(*, lm_loss_impl: str) -> MetisMambaConfig:
    return MetisMambaConfig(
        name="Metis-1.5-contract-smoke",
        vocab_size=128,
        block_size=8,
        d_model=64,
        n_layer=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        tie_embeddings=False,
        attention_backend="sdpa",
        training_mode="static_dense_pretrain",
        mor_enabled=False,
        mor_runtime_mode="disabled",
        ffn_type="swiglu",
        low_precision_mode="none",
        lm_loss_impl=lm_loss_impl,
    )


def tiny_single_latent_config() -> MetisMambaConfig:
    return MetisMambaConfig(
        name="Metis-1.5-single-latent-smoke",
        vocab_size=128,
        block_size=16,
        d_model=64,
        n_layer=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        tie_embeddings=False,
        attention_backend="sdpa",
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
        moe_balance_strategy="aux_loss_free_bias",
        moe_balance_bias_update_rate=1e-3,
        moe_aux_loss_coef=1e-4,
        moe_backend="te_grouped",
        moe_dispatch_mode="grouped",
        low_precision_mode="none",
        lm_loss_impl="standard",
    )


def assert_single_latent_moe_contract(device: torch.device) -> None:
    torch.manual_seed(20260514)
    config = tiny_single_latent_config()
    audit = config.param_application_audit()
    if audit["routing_units_per_token"] != config.moe_top_k:
        raise AssertionError("single_latent_moe must route exactly top_k experts per token.")
    model = build_model(config, device=device, dtype=torch.float32, use_fp8=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size), device=device)

    for _step in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=input_ids, return_logits=False)
        if output.loss is None or not torch.isfinite(output.loss):
            raise AssertionError("Expected finite single_latent_moe training loss.")
        output.loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise AssertionError(f"Nonfinite gradient in {name}.")
        optimizer.step()
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise AssertionError(f"Nonfinite parameter in {name}.")

    counters = model.get_perf_counters()
    expected_assignments = 5 * config.n_layer * input_ids.numel() * config.moe_top_k
    if counters.get("moe_grouped_assignments", 0) != expected_assignments:
        raise AssertionError(
            "single_latent_moe dispatch count mismatch: "
            f"got {counters.get('moe_grouped_assignments', 0)}, expected {expected_assignments}."
        )
    print(
        "single_latent_moe_training_ok "
        f"loss={float(output.loss.detach()):.6f} assignments={counters.get('moe_grouped_assignments', 0)} "
        f"rough_param_apps={audit['rough_total_param_apps_per_token']:,}",
        flush=True,
    )


def assert_force_balanced_router_contract(device: torch.device) -> None:
    torch.manual_seed(20260526)
    config = tiny_single_latent_config()
    config.n_layer = 1
    config.moe_router_override = "force_balanced"
    config.moe_backend = "torch_looped"
    config.moe_dispatch_mode = "grouped"
    config.validate()
    model = build_model(config, device=device, dtype=torch.float32, use_fp8=False)
    model.train()
    mlp = model.backbone.layers[0].mlp
    latent = torch.randn(2, config.block_size, config.moe_routed_latent_size, device=device)
    topk_indices, topk_weights, aux_loss, _router_logits = mlp._route_latents(
        latent,
        is_first_microbatch=True,
    )
    counts = torch.bincount(topk_indices.reshape(-1), minlength=config.moe_num_experts)
    expected = topk_indices.numel() // config.moe_num_experts
    if not torch.equal(counts, torch.full_like(counts, expected)):
        raise AssertionError(f"force-balanced router counts mismatch: {counts.tolist()} vs expected {expected}.")
    torch.testing.assert_close(
        topk_weights,
        torch.full_like(topk_weights, 1.0 / float(config.moe_top_k)),
        rtol=0.0,
        atol=0.0,
    )
    if float(aux_loss.detach()) != 0.0:
        raise AssertionError("force-balanced router override should not add learned-router aux loss.")
    input_ids = torch.randint(0, config.vocab_size, (2, config.block_size), device=device)
    output = model(input_ids, labels=input_ids, return_logits=False)
    if output.loss is None or not torch.isfinite(output.loss):
        raise AssertionError("Expected finite force-balanced single_latent_moe loss.")
    output.loss.backward()
    print(
        "force_balanced_router_ok "
        f"counts={counts.tolist()} loss={float(output.loss.detach()):.6f}",
        flush=True,
    )


@torch.no_grad()
def assert_next_token_contract(device: torch.device, *, check_liger: bool) -> None:
    torch.manual_seed(20260513)
    config = tiny_config(lm_loss_impl="standard")
    model = build_model(config, device=device, dtype=torch.float32, use_fp8=False)
    model.eval()

    input_ids = torch.tensor(
        [
            [3, 7, 11, 19, 23, 31, 43, 59],
            [5, 13, 17, 29, 37, 41, 53, 61],
        ],
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    output = model(input_ids, labels=labels, return_logits=True)
    if output.logits is None or output.lm_loss is None:
        raise AssertionError("Expected logits and language-model loss from standard CE path.")

    next_token_loss = F.cross_entropy(
        output.logits[:, :-1, :].contiguous().view(-1, output.logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    two_token_loss = F.cross_entropy(
        output.logits[:, :-2, :].contiguous().view(-1, output.logits.size(-1)),
        labels[:, 2:].contiguous().view(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(output.lm_loss, next_token_loss, rtol=0.0, atol=1e-6)
    if torch.allclose(output.lm_loss, two_token_loss, rtol=0.0, atol=1e-6):
        raise AssertionError("Standard CE loss unexpectedly matched the two-token-ahead objective.")
    print(
        "standard_ce_next_token_ok "
        f"loss={float(output.lm_loss):.6f} two_token_loss={float(two_token_loss):.6f}",
        flush=True,
    )

    if not check_liger:
        return
    if device.type != "cuda":
        raise RuntimeError("--check-liger requires a CUDA device.")
    liger_config = copy.deepcopy(config)
    liger_config.lm_loss_impl = "liger_fused_linear_ce"
    liger_model = build_model(liger_config, device=device, dtype=torch.bfloat16, use_fp8=False)
    liger_model.load_state_dict(model.state_dict())
    liger_model.eval()
    liger_output = liger_model(input_ids, labels=labels, return_logits=True)
    if liger_output.logits is None or liger_output.lm_loss is None:
        raise AssertionError("Expected logits and language-model loss from Liger CE path.")
    liger_manual = F.cross_entropy(
        liger_output.logits[:, :-1, :].float().contiguous().view(-1, liger_output.logits.size(-1)),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    torch.testing.assert_close(liger_output.lm_loss.float(), liger_manual, rtol=2e-2, atol=2e-2)
    print(
        "liger_fused_linear_ce_next_token_ok "
        f"loss={float(liger_output.lm_loss):.6f} manual={float(liger_manual):.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-liger", action="store_true")
    args = parser.parse_args()
    assert_next_token_contract(torch.device(args.device), check_liger=args.check_liger)
    assert_single_latent_moe_contract(torch.device(args.device))
    assert_force_balanced_router_contract(torch.device(args.device))


if __name__ == "__main__":
    main()
