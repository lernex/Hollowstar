#!/usr/bin/env python3
"""Reproduce the Eventide working neural-parameter ledger.

This deliberately counts learned parameters separately from runtime state and
counts selected expert-embedding rows, rather than their whole stored table, in
the active-per-pass budget.  The architecture is still a research target; this
script exists so changes to one dimension cannot silently leave the prose with
an arithmetically incompatible 50B/A2B claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EventideDimensions:
    d_model: int = 2_048
    d_latent: int = 512
    n_blocks: int = 32
    n_mamba: int = 28
    n_hca: int = 2
    n_csa: int = 2
    n_streams: int = 4
    max_passes: int = 5
    pass_embedding_dim: int = 64
    route_feature_dim: int = 256
    memory_dim: int = 256
    vocab_size: int = 131_072
    n_experts: int = 1_024
    expert_ffn: int = 960
    max_k: int = 32
    shared_ffn: int = 1_536
    hca_csa_head_dim: int = 256
    hca_csa_query_heads: int = 64
    hca_csa_query_rank: int = 512
    hca_csa_output_groups: int = 8
    hca_csa_output_rank: int = 512
    csa_compress_rate: int = 4
    hca_compress_rate: int = 128
    index_heads: int = 32
    index_head_dim: int = 128
    first_attention_layer_index: int = 7
    partial_rope_dim: int = 32
    local_attention_window: int = 128
    main_cache_scale_metadata_bytes: int = 2
    index_cache_scale_group_size: int = 16
    mamba_expand: int = 2
    mamba_head_dim: int = 64
    mamba_state_dim: int = 128
    mamba_conv_kernel: int = 4
    mamba_state_value_bytes: int = 2
    ngram_orders: int = 2
    ngram_tables_per_order: int = 8
    ngram_value_dim: int = 64
    ngram_injection_sites: int = 4
    ngram_logical_parameters: int = 200_000_000_000


def calculate(d: EventideDimensions = EventideDimensions()) -> dict[str, object]:
    # Official Mamba-3 SISO defaults at d_model=2048: expand=2, d_state=128,
    # headdim=64, ngroups=1, d_conv=4. This includes its learned internal norm.
    mamba_per_block = 26_165_632

    c = d.hca_csa_head_dim
    nh = d.hca_csa_query_heads
    dc = d.hca_csa_query_rank
    g = d.hca_csa_output_groups
    dg = d.hca_csa_output_rank
    core_attention = (
        d.d_model * dc
        + dc
        + dc * nh * c
        + d.d_model * c
        + c
        + (nh * c // g) * (g * dg)
        + (g * dg) * d.d_model
        + nh
    )
    hca_compressor = (
        d.d_model * c
        + d.d_model * c
        + d.hca_compress_rate * c
        + c
    )
    csa_outer_compressor = (
        d.d_model * (2 * c)
        + d.d_model * (2 * c)
        + d.csa_compress_rate * (2 * c)
        + c
    )
    ci = d.index_head_dim
    nhi = d.index_heads
    csa_indexer = (
        d.d_model * (2 * ci)
        + d.d_model * (2 * ci)
        + d.csa_compress_rate * (2 * ci)
        + ci
        + dc * nhi * ci
        + d.d_model * nhi
    )
    hca_per_block = core_attention + hca_compressor
    csa_per_block = core_attention + csa_outer_compressor + csa_indexer

    # Two DeepSeek-style dynamic mHC sites per block, with a small additive
    # pass-conditioned projection so the recurrent application is identifiable.
    mhc_outputs = (2 + d.n_streams) * d.n_streams
    mhc_per_connection = (
        mhc_outputs * (d.n_streams * d.d_model)
        + mhc_outputs
        + 3
        + d.pass_embedding_dim * mhc_outputs
        + mhc_outputs
    )
    mhc_connections = d.n_blocks * 2 * mhc_per_connection
    mhc = (
        mhc_connections
        + d.max_passes * d.pass_embedding_dim
        + d.n_streams * d.d_model
    )

    route_input = d.d_latent + d.route_feature_dim
    expert_router = d.n_blocks * (route_input * d.n_experts + d.n_experts)
    k_choices = d.max_k + 1
    k_router = d.n_blocks * (route_input * k_choices + k_choices)
    expert_route_embeddings = d.n_blocks * d.n_experts * d.route_feature_dim

    metadata_width = d.route_feature_dim + 4
    n_attention = d.n_hca + d.n_csa
    memory_detail = {
        "state_write": d.d_model * d.memory_dim + d.memory_dim,
        "metadata_write": metadata_width * d.memory_dim + d.memory_dim,
        "pass_embeddings": d.max_passes * d.memory_dim,
        "anchor_embeddings": (n_attention + 1) * d.memory_dim,
        "query": d.d_model * d.memory_dim + d.memory_dim,
        "key": d.memory_dim * d.memory_dim + d.memory_dim,
        "value": d.memory_dim * d.memory_dim + d.memory_dim,
        "output": d.memory_dim * d.d_model + d.d_model,
        "route_projection": (3 * d.d_model + d.route_feature_dim)
        * d.route_feature_dim
        + d.route_feature_dim,
        "stream_gate": d.n_streams * d.d_model + d.n_streams,
    }
    depth_memory = sum(memory_detail.values())
    continuation_input = 3 * d.d_model + d.route_feature_dim
    continuation_detail = {
        "hidden": continuation_input * d.route_feature_dim
        + d.route_feature_dim,
        "output": d.route_feature_dim + 1,
    }
    continuation = sum(continuation_detail.values())

    ngram_retrieved_rows = d.ngram_orders * d.ngram_tables_per_order
    ngram_concatenated_dim = ngram_retrieved_rows * d.ngram_value_dim
    ngram_projection = ngram_concatenated_dim * d.d_model + d.d_model
    ngram_injection_gates = (
        d.ngram_injection_sites
        * d.max_passes
        * d.n_streams
        * (d.d_model + 1)
    )
    ngram_active_gates_per_pass = (
        d.ngram_injection_sites * d.n_streams * (d.d_model + 1)
    )

    stored = {
        "routed_expert_bank": d.n_blocks
        * d.n_experts
        * 3
        * d.d_latent
        * d.expert_ffn,
        "shared_experts": d.n_blocks * 3 * d.d_model * d.shared_ffn,
        "mamba3_siso": d.n_mamba * mamba_per_block,
        "hca": d.n_hca * hca_per_block,
        "csa": d.n_csa * csa_per_block,
        "latent_down_up": d.n_blocks * 2 * d.d_model * d.d_latent,
        "expert_router": expert_router,
        "k_router": k_router,
        "expert_route_embeddings": expert_route_embeddings,
        "mhc": mhc,
        "depth_memory": depth_memory,
        "continuation_controller": continuation,
        "ngram_projection": ngram_projection,
        "ngram_injection_gates": ngram_injection_gates,
        "block_norms": d.n_blocks * 2 * d.d_model,
        "token_embedding": d.vocab_size * d.d_model,
        "final_norm": d.d_model,
    }
    untied_head = d.vocab_size * d.d_model

    # Expert identities are computed by the dense router, but only selected
    # route-embedding rows are read. Their K-dependent row reads join the much
    # larger selected expert matrices in the slope.
    # A pass reads one pass-embedding row; the stream seeds are applied once
    # when the token enters the recurrent body, not once on every pass.
    mhc_active_per_pass = mhc_connections + d.pass_embedding_dim

    memory_read_projection = (
        d.d_model * d.memory_dim
        + d.memory_dim
        + d.memory_dim * d.memory_dim
        + d.memory_dim
        + d.memory_dim * d.memory_dim
        + d.memory_dim
        + d.memory_dim * d.d_model
        + d.d_model
        + (3 * d.d_model + d.route_feature_dim) * d.route_feature_dim
        + d.route_feature_dim
        + d.n_streams * d.d_model
        + d.n_streams
    )
    memory_write_projection = (
        d.d_model * d.memory_dim
        + d.memory_dim
        + metadata_width * d.memory_dim
        + d.memory_dim
    )
    memory_active_unique = (
        memory_read_projection
        + memory_write_projection
        + d.memory_dim  # selected pass row
        + (n_attention + 1) * d.memory_dim  # all anchor rows
    )

    active_fixed = (
        stored["shared_experts"]
        + stored["mamba3_siso"]
        + stored["hca"]
        + stored["csa"]
        + stored["latent_down_up"]
        + stored["expert_router"]
        + stored["k_router"]
        + mhc_active_per_pass
        + memory_active_unique
        + stored["continuation_controller"]
        + ngram_active_gates_per_pass
        + stored["block_norms"]
    )
    active_per_k = (
        d.n_blocks * 3 * d.d_latent * d.expert_ffn
        + d.n_blocks * d.route_feature_dim
    )
    mean_k_for_a2b = (2_000_000_000 - active_fixed) / active_per_k
    active = {
        "fixed_per_pass": active_fixed,
        "per_1_k": active_per_k,
        "mean_k_for_a2b": mean_k_for_a2b,
        "k0_per_pass": active_fixed,
        "mean_per_pass": active_fixed + mean_k_for_a2b * active_per_k,
        "kmax_per_pass": active_fixed + d.max_k * active_per_k,
        "mhc_unique_active_per_pass": mhc_active_per_pass,
        "depth_memory_unique_active_per_pass": memory_active_unique,
    }

    # The recurrent memory owns one shared set of weights but invokes it many
    # times inside a pass. The first pass has no memory to retrieve before the
    # first HCA site (after seven Mamba blocks), while every later pass starts
    # with a populated bank. Route-feature projection still runs for those
    # early empty-bank blocks.
    retrieve_without_route = memory_read_projection - (
        (3 * d.d_model + d.route_feature_dim) * d.route_feature_dim
        + d.route_feature_dim
    )
    route_projection = memory_read_projection - retrieve_without_route
    reads_first_pass = d.n_blocks - d.first_attention_layer_index
    reads_later_pass = d.n_blocks + 1
    writes_per_pass = n_attention + 1
    memory_write_applications = writes_per_pass * (
        memory_write_projection + 2 * d.memory_dim
    )
    memory_applications_first_pass = (
        reads_first_pass * retrieve_without_route
        + reads_later_pass * route_projection
        + memory_write_applications
    )
    memory_applications_later_pass = (
        reads_later_pass * memory_read_projection + memory_write_applications
    )
    once_per_output_interface = (
        untied_head
        + d.d_model  # one token-embedding row
        + d.d_model  # final norm
        + d.n_streams * d.d_model  # stream seeds
        + ngram_projection
        + ngram_concatenated_dim  # 16 selected table rows, once per token
    )
    applications = {
        "depth_memory_first_pass": memory_applications_first_pass,
        "depth_memory_later_pass": memory_applications_later_pass,
        "mean_depth_2_body": (
            2 * active["mean_per_pass"]
            + memory_applications_first_pass
            + memory_applications_later_pass
            - 2 * memory_active_unique
        ),
        "max_depth_5_body": (
            d.max_passes * active["kmax_per_pass"]
            + memory_applications_first_pass
            + (d.max_passes - 1) * memory_applications_later_pass
            - d.max_passes * memory_active_unique
        ),
        "once_per_output_interface": once_per_output_interface,
    }
    applications["mean_depth_2_head_inclusive"] = (
        applications["mean_depth_2_body"] + once_per_output_interface
    )
    applications["max_depth_5_head_inclusive"] = (
        applications["max_depth_5_body"] + once_per_output_interface
    )

    runtime_buffers = {
        "selection_bias_fp32_bytes": d.n_blocks * d.n_experts * 4,
        "k_budget_multiplier_fp32_bytes": d.n_blocks * 4,
        "depth_budget_multiplier_fp32_bytes": 4,
    }
    ngram_rows = d.ngram_logical_parameters // d.ngram_value_dim
    ngram_storage = {
        "logical_parameters": d.ngram_logical_parameters,
        "retrieved_values_per_token": ngram_concatenated_dim,
        "radix3_payload_bytes": d.ngram_logical_parameters * 32 // 20 // 8,
        "bf16_row_scale_bytes": ngram_rows * 2,
    }
    ngram_storage["radix3_plus_row_scales_bytes"] = (
        ngram_storage["radix3_payload_bytes"]
        + ngram_storage["bf16_row_scale_bytes"]
    )

    totals = {
        "stored_tied": sum(stored.values()),
        "stored_untied": sum(stored.values()) + untied_head,
        "untied_head": untied_head,
    }

    # Decode-residency estimate for the standard B32 x 262K target. Main
    # compressed-KV rows keep 32 partial-RoPE dimensions in BF16, the other
    # 224 in FP8, plus two bytes of scale metadata. Index rows use FP4 with
    # one byte of scale per 16 values.
    batch = 32
    context = 262_144
    main_cache_row_bytes = (
        (c - d.partial_rope_dim)
        + 2 * d.partial_rope_dim
        + d.main_cache_scale_metadata_bytes
    )
    index_cache_row_bytes = (
        ci // 2 + ci // d.index_cache_scale_group_size
    )
    attention_cache_bytes = (
        d.n_csa * batch * (context // d.csa_compress_rate) * main_cache_row_bytes
        + d.n_csa * batch * (context // d.csa_compress_rate) * index_cache_row_bytes
        + d.n_hca * batch * (context // d.hca_compress_rate) * main_cache_row_bytes
        + (d.n_hca + d.n_csa)
        * batch
        * d.local_attention_window
        * main_cache_row_bytes
    )
    mamba_inner = d.d_model * d.mamba_expand
    mamba_heads = mamba_inner // d.mamba_head_dim
    mamba_ssm_state_bytes = (
        d.n_mamba
        * batch
        * mamba_heads
        * d.mamba_head_dim
        * d.mamba_state_dim
        * d.mamba_state_value_bytes
    )
    mamba_conv_state_bytes = (
        d.n_mamba
        * batch
        * (mamba_inner + 2 * d.mamba_state_dim)
        * d.mamba_conv_kernel
        * d.mamba_state_value_bytes
    )
    # 1.725 bpw = radix-3 payload plus one BF16 scale per 128 weights.
    # The mixed scenario deliberately reserves 1% of parameters for NVFP4
    # and 1% for FP8 rather than pretending all validation islands are known.
    all_ternary_payload_bytes = totals["stored_untied"] * 1.6 / 8
    all_ternary_scaled_bytes = totals["stored_untied"] * 1.725 / 8
    conservative_mixed_weight_bytes = totals["stored_untied"] * (
        0.98 * 1.725 + 0.01 * 4.5 + 0.01 * 8.0
    ) / 8
    sequence_state_bytes = (
        attention_cache_bytes + mamba_ssm_state_bytes + mamba_conv_state_bytes
    )
    deployment = {
        "batch": batch,
        "context_per_sequence": context,
        "main_cache_row_bytes": main_cache_row_bytes,
        "index_cache_row_bytes": index_cache_row_bytes,
        "all_ternary_payload_gib": all_ternary_payload_bytes / (1 << 30),
        "all_ternary_with_bf16_scale_per_128_gib": all_ternary_scaled_bytes
        / (1 << 30),
        "98pct_ternary_1pct_nvfp4_1pct_fp8_gib": conservative_mixed_weight_bytes
        / (1 << 30),
        "attention_cache_gib": attention_cache_bytes / (1 << 30),
        "mamba_state_gib": (mamba_ssm_state_bytes + mamba_conv_state_bytes)
        / (1 << 30),
        "sequence_state_gib": sequence_state_bytes / (1 << 30),
        "gib_left_after_conservative_weights_and_sequence_state": 16
        - (conservative_mixed_weight_bytes + sequence_state_bytes) / (1 << 30),
        "b16_sequence_state_gib": sequence_state_bytes / 2 / (1 << 30),
        "b16_gib_left_after_conservative_weights_and_sequence_state": 16
        - (conservative_mixed_weight_bytes + sequence_state_bytes / 2)
        / (1 << 30),
    }
    control_experts = d.n_experts // 2
    control_ffn = d.expert_ffn * 2
    control_max_k = d.max_k // 2
    control_expert_router = d.n_blocks * (
        route_input * control_experts + control_experts
    )
    control_k_choices = control_max_k + 1
    control_k_router = d.n_blocks * (
        route_input * control_k_choices + control_k_choices
    )
    control_route_embeddings = (
        d.n_blocks * control_experts * d.route_feature_dim
    )
    control_fixed = (
        active_fixed
        - expert_router
        - k_router
        + control_expert_router
        + control_k_router
    )
    control_per_k = (
        d.n_blocks * 3 * d.d_latent * control_ffn
        + d.n_blocks * d.route_feature_dim
    )
    control_capacity_matched_k = mean_k_for_a2b / 2
    control = {
        "n_experts": control_experts,
        "expert_ffn": control_ffn,
        "max_k": control_max_k,
        "stored_untied": (
            totals["stored_untied"]
            - expert_router
            - k_router
            - expert_route_embeddings
            + control_expert_router
            + control_k_router
            + control_route_embeddings
        ),
        "capacity_matched_mean_k": control_capacity_matched_k,
        "capacity_matched_active_per_pass": control_fixed
        + control_capacity_matched_k * control_per_k,
        "a2b_mean_k": (2_000_000_000 - control_fixed) / control_per_k,
    }

    assert hca_per_block == 27_821_120
    assert csa_per_block == 32_051_392
    assert mhc == 12_692_992
    assert depth_memory == 3_423_236
    assert continuation == 1_638_913
    assert ngram_projection == 2_099_200
    assert ngram_injection_gates == 163_920
    assert totals["stored_untied"] == 50_131_285_109
    assert active["mean_per_pass"] == 2_000_000_000
    assert memory_active_unique == 3_422_212
    assert memory_applications_first_pass == 86_801_508
    assert memory_applications_later_pass == 96_326_788
    assert control["stored_untied"] == 50_114_097_781

    return {
        "dimensions": asdict(d),
        "attention_detail": {
            "core_per_block": core_attention,
            "hca_compressor_per_block": hca_compressor,
            "csa_outer_compressor_per_block": csa_outer_compressor,
            "csa_indexer_per_block": csa_indexer,
            "hca_per_block": hca_per_block,
            "csa_per_block": csa_per_block,
        },
        "router_detail": {
            "expert_router_per_block": expert_router // d.n_blocks,
            "k_router_per_block": k_router // d.n_blocks,
            "expert_route_embeddings_per_block": expert_route_embeddings
            // d.n_blocks,
        },
        "mhc_per_connection": mhc_per_connection,
        "depth_memory_detail": memory_detail,
        "continuation_detail": continuation_detail,
        "stored_parameters": stored,
        "active_parameters": active,
        "parameter_applications": applications,
        "runtime_nonparameter_buffers": runtime_buffers,
        "external_ngram_memory": ngram_storage,
        "b32_262k_decode_residency_estimate": deployment,
        "totals": totals,
        "matched_512_expert_control": control,
    }


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2, sort_keys=True))
