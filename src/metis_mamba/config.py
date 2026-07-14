from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MetisMambaConfig:
    name: str = "Metis-1.5"
    model_type: str = "metis_mor_transformer"
    architecture: str = "metis_mor_decoder"
    vocab_size: int = 16384
    block_size: int = 1024
    d_model: int = 1536
    n_layer: int = 19
    n_heads: int = 24
    n_kv_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 4096
    hidden_act: str = "swiglu"
    tie_embeddings: bool = True
    rms_norm: bool = True
    residual_in_fp32: bool = False
    fused_add_norm: bool = False
    pad_vocab_size_multiple: int = 16
    initializer_range: float = 0.02
    torch_dtype: str = "bfloat16"
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0
    # "neox" is the correct half-split rotation (matches the JAX lane; required
    # for Metis-1.5 weights). "legacy_metis14" preserves the historical mixed
    # convention that Metis-1.4 checkpoints were trained with.
    rope_convention: str = "legacy_metis14"
    attention_backend: str = "auto"
    training_mode: str = "dynamic_token_mor"
    mor_enabled: bool = True
    mor_train_router: bool = True
    mor_runtime_mode: str = "dynamic_token"
    attention_mask_mode: str = "auto"
    disable_depth_stack: bool = False
    disable_token_packing: bool = False
    disable_token_scatter: bool = False
    debug_attention_backend: bool = False
    debug_perf_counters: bool = False
    native_gqa_attention: bool = True
    te_dot_product_attention: bool = False
    cuda_graphs: bool = False
    cuda_graph_scope: str = "none"
    fp8_dpa: bool = False
    fp8_mha: bool = False
    fp8_expert_precision: str = "fp8"
    low_precision_mode: str = "none"
    nvfp4_disable_rht: bool = False
    nvfp4_disable_2d_quantization: bool = False
    nvfp4_disable_stochastic_rounding: bool = False
    nvfp4_keep_embeddings_bf16: bool = True
    nvfp4_keep_qkv_bf16: bool = True
    nvfp4_keep_latent_moe_projections_bf16: bool = True
    nvfp4_keep_lm_head_bf16: bool = True
    nvfp4_qkv_precision: str = "legacy"
    nvfp4_latent_moe_projection_precision: str = "legacy"
    nvfp4_lm_head_precision: str = "legacy"
    te_fused_mlp: bool = False
    lm_loss_impl: str = "standard"
    ffn_type: str = "swiglu"
    moe_num_experts: int = 0
    moe_top_k: int = 0
    moe_shared_experts: int = 0
    moe_num_heads: int = 1
    moe_expert_intermediate_size: int = 0
    moe_router_latent_size: int = 0
    moe_routed_latent_size: int = 0
    moe_activation: str = "swiglu"
    moe_router_temperature: float = 1.0
    moe_aux_loss_coef: float = 0.0
    moe_router_score: str = "sigmoid"
    moe_router_override: str = "learned"
    moe_single_latent_router_input: str = "hidden"
    moe_balance_strategy: str = "aux_loss"
    moe_balance_bias_update_rate: float = 0.0
    moe_balance_bias_clamp: float = 5.0
    moe_balance_scale_by_token_fraction: bool = True
    moe_backend: str = "te_grouped"
    moe_dispatch_mode: str = "grouped"
    moe_static_capacity: int = 0
    moe_capacity_factor: float = 0.0
    moe_capacity_alignment: int = 128
    moe_torch_grouped_min_m: int = 0
    moe_overflow_mode: str = "fallback"
    moe_fused_combine: bool = True
    moe_graphable: bool = False
    moe_memory_efficient_permutation: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    moe_flex_dispatcher_backend: str = "none"
    moe_overlap_expert_parallel_comm: bool = False
    moe_delay_wgrad_compute: bool = False
    moe_router_fusion: bool = False
    moe_permute_fusion: bool = True
    moe_expert_parallel_size: int = 1
    moe_expert_parallel_init_seed: int = 271828
    nvfp4_final_expert_layers: int = 0
    nvfp4_final_expert_precision: str = "fp8_block"
    mor_max_depth: int = 3
    mor_router_hidden_dim: int = 256
    mor_router_temperature: float = 1.0
    mor_router_aux_loss_coef: float = 0.01
    mor_router_entropy_coef: float = 0.0
    mor_router_z_loss_coef: float = 0.0
    mor_target_avg_depth: float = 1.5
    mor_depth2_capacity_sequences: int = 0
    mor_depth3_capacity_sequences: int = 0
    mor_block_size: int = 128
    mor_depth2_capacity_blocks: int = 0
    mor_depth3_capacity_blocks: int = 0
    block_mor_attention_mode: str = "local_block_refinement"
    fp8_pad_multiple: int = 64

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetisMambaConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        cooked = {key: value for key, value in payload.items() if key in allowed}
        return cls(**cooked)

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.d_model <= 0 or self.n_layer <= 0:
            raise ValueError("d_model and n_layer must be positive.")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("n_heads and n_kv_heads must be positive.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model.")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")
        if self.hidden_act not in {"swiglu"}:
            raise ValueError("hidden_act must currently be swiglu.")
        if self.pad_vocab_size_multiple <= 0:
            raise ValueError("pad_vocab_size_multiple must be positive.")
        if self.attention_backend not in {"auto", "flash_attention_3", "sdpa", "eager"}:
            raise ValueError("attention_backend must be one of: auto, flash_attention_3, sdpa, eager.")
        if self.te_dot_product_attention and self.attention_backend == "eager":
            raise ValueError("te_dot_product_attention cannot be enabled with attention_backend='eager'.")
        if self.training_mode not in {
            "dynamic_token_mor",
            "static_dense_pretrain",
            "static_sequence_mor",
            "static_block_mor",
        }:
            raise ValueError(
                "training_mode must be one of: dynamic_token_mor, static_dense_pretrain, "
                "static_sequence_mor, static_block_mor."
            )
        if self.mor_runtime_mode not in {"dynamic_token", "disabled", "static_sequence", "static_block"}:
            raise ValueError("mor_runtime_mode must be one of: dynamic_token, disabled, static_sequence, static_block.")
        if self.attention_mask_mode not in {"auto", "causal_none"}:
            raise ValueError("attention_mask_mode must be one of: auto, causal_none.")
        if self.cuda_graph_scope not in {"none", "step", "microbatch"}:
            raise ValueError("cuda_graph_scope must be one of: none, step, microbatch.")
        if self.lm_loss_impl not in {"standard", "liger_fused_linear_ce"}:
            raise ValueError("lm_loss_impl must be one of: standard, liger_fused_linear_ce.")
        if self.low_precision_mode not in {"none", "fp8", "nvfp4"}:
            raise ValueError("low_precision_mode must be one of: none, fp8, nvfp4.")
        if self.fp8_expert_precision not in {"fp8", "bf16"}:
            raise ValueError("fp8_expert_precision must be one of: fp8, bf16.")
        valid_nvfp4_precisions = {"legacy", "bf16", "mxfp8", "fp8_block", "nvfp4"}
        if self.nvfp4_qkv_precision not in valid_nvfp4_precisions:
            raise ValueError("nvfp4_qkv_precision must be one of: legacy, bf16, mxfp8, fp8_block, nvfp4.")
        if self.nvfp4_latent_moe_projection_precision not in valid_nvfp4_precisions:
            raise ValueError(
                "nvfp4_latent_moe_projection_precision must be one of: legacy, bf16, mxfp8, fp8_block, nvfp4."
            )
        if self.nvfp4_lm_head_precision not in valid_nvfp4_precisions:
            raise ValueError("nvfp4_lm_head_precision must be one of: legacy, bf16, mxfp8, fp8_block, nvfp4.")
        if self.nvfp4_final_expert_precision not in {"bf16", "mxfp8", "fp8_block", "nvfp4"}:
            raise ValueError("nvfp4_final_expert_precision must be one of: bf16, mxfp8, fp8_block, nvfp4.")
        if self.nvfp4_final_expert_layers < 0:
            raise ValueError("nvfp4_final_expert_layers cannot be negative.")
        if self.ffn_type not in {"swiglu", "multi_head_latent_moe", "single_latent_moe"}:
            raise ValueError("ffn_type must be one of: swiglu, multi_head_latent_moe, single_latent_moe.")
        if self.uses_moe:
            if self.te_fused_mlp:
                raise ValueError("te_fused_mlp cannot be enabled with MoE FFN types.")
            if self.moe_num_experts <= 0:
                raise ValueError("moe_num_experts must be positive for MoE FFN types.")
            if self.moe_top_k <= 0 or self.moe_top_k > self.moe_num_experts:
                raise ValueError("moe_top_k must be within [1, moe_num_experts].")
            if self.moe_shared_experts < 0:
                raise ValueError("moe_shared_experts cannot be negative.")
            if self.ffn_type == "multi_head_latent_moe":
                if self.moe_num_heads <= 0 or self.d_model % self.moe_num_heads != 0:
                    raise ValueError("d_model must be divisible by moe_num_heads.")
            elif self.moe_num_heads != 1:
                raise ValueError("single_latent_moe must use moe_num_heads=1.")
            if self.moe_expert_intermediate_size <= 0:
                raise ValueError("moe_expert_intermediate_size must be positive for MoE FFN types.")
            if self.moe_router_latent_size <= 0:
                raise ValueError("moe_router_latent_size must be positive for MoE FFN types.")
            if self.moe_routed_latent_size < 0:
                raise ValueError("moe_routed_latent_size cannot be negative.")
            if self.ffn_type == "multi_head_latent_moe":
                if self.moe_routed_latent_size > 0 and self.moe_routed_latent_size > self.moe_head_dim:
                    raise ValueError("moe_routed_latent_size should not exceed moe_head_dim.")
            elif self.moe_routed_latent_size <= 0:
                raise ValueError("single_latent_moe requires moe_routed_latent_size to define latent_dim.")
            elif self.moe_routed_latent_size > self.d_model:
                raise ValueError("single_latent_moe latent_dim must not exceed d_model.")
            if self.moe_activation not in {"swiglu", "squared_relu"}:
                raise ValueError("moe_activation must be one of: swiglu, squared_relu.")
            if self.moe_router_temperature <= 0:
                raise ValueError("moe_router_temperature must be positive.")
            if self.moe_aux_loss_coef < 0:
                raise ValueError("moe_aux_loss_coef cannot be negative.")
            if self.moe_router_score not in {"sigmoid", "softmax"}:
                raise ValueError("moe_router_score must be one of: sigmoid, softmax.")
            if self.moe_router_override not in {"learned", "force_balanced", "uniform_random"}:
                raise ValueError("moe_router_override must be one of: learned, force_balanced, uniform_random.")
            if self.moe_single_latent_router_input not in {"hidden", "latent"}:
                raise ValueError("moe_single_latent_router_input must be one of: hidden, latent.")
            if self.moe_balance_strategy not in {"aux_loss", "aux_loss_free_bias"}:
                raise ValueError("moe_balance_strategy must be one of: aux_loss, aux_loss_free_bias.")
            if self.moe_balance_bias_update_rate < 0:
                raise ValueError("moe_balance_bias_update_rate cannot be negative.")
            if self.moe_balance_bias_clamp <= 0:
                raise ValueError("moe_balance_bias_clamp must be positive.")
            if self.moe_backend not in {
                "te_grouped",
                "torch_grouped",
                "torch_grouped_safe",
                "torch_bmm",
                "torch_looped",
                "cudnnfe",
                "triton",
                "cutlass",
            }:
                raise ValueError(
                    "moe_backend must be one of: te_grouped, torch_grouped, torch_grouped_safe, "
                    "torch_bmm, torch_looped, cudnnfe, triton, cutlass."
                )
            if self.moe_backend not in {"te_grouped", "torch_grouped", "torch_grouped_safe", "torch_bmm", "torch_looped"}:
                raise ValueError(
                    "Only moe_backend='te_grouped', 'torch_grouped', 'torch_grouped_safe', 'torch_bmm', and "
                    "'torch_looped' "
                    "are wired in this checkout. "
                    "Use moe_dispatch_mode='bucketed' for the Triton dispatcher while the "
                    "cuDNN FE/CUTLASS grouped-GEMM backend is implemented."
                )
            if self.moe_dispatch_mode not in {"loop", "sorted_loop", "grouped", "bucketed"}:
                raise ValueError("moe_dispatch_mode must be one of: loop, sorted_loop, grouped, bucketed.")
            if self.moe_static_capacity < 0:
                raise ValueError("moe_static_capacity cannot be negative.")
            if self.moe_capacity_factor < 0.0:
                raise ValueError("moe_capacity_factor cannot be negative.")
            if self.moe_capacity_alignment <= 0:
                raise ValueError("moe_capacity_alignment must be positive.")
            if self.moe_torch_grouped_min_m < 0:
                raise ValueError("moe_torch_grouped_min_m cannot be negative.")
            if self.moe_overflow_mode not in {"fallback", "drop", "error"}:
                raise ValueError("moe_overflow_mode must be one of: fallback, drop, error.")
            if self.moe_graphable and self.moe_dispatch_mode != "bucketed":
                raise ValueError("moe_graphable requires moe_dispatch_mode='bucketed'.")
            if self.moe_graphable and self.moe_static_capacity <= 0 and self.moe_capacity_factor <= 0.0:
                raise ValueError("moe_graphable requires moe_static_capacity > 0 or moe_capacity_factor > 0.")
            if self.moe_graphable and self.moe_overflow_mode != "drop":
                raise ValueError("moe_graphable requires moe_overflow_mode='drop' to avoid host overflow checks.")
            if self.moe_memory_efficient_permutation and self.mlp_bias:
                raise ValueError("moe_memory_efficient_permutation requires mlp_bias=false for exact expert math.")
            if self.moe_token_dispatcher_type not in {"alltoall", "flex"}:
                raise ValueError("moe_token_dispatcher_type must be one of: alltoall, flex.")
            if self.moe_flex_dispatcher_backend not in {"none", "deepep", "hybridep", "auto"}:
                raise ValueError("moe_flex_dispatcher_backend must be one of: none, deepep, hybridep, auto.")
            if self.moe_token_dispatcher_type == "flex" and self.moe_flex_dispatcher_backend == "none":
                raise ValueError("moe_token_dispatcher_type='flex' requires moe_flex_dispatcher_backend.")
            if self.moe_token_dispatcher_type == "flex":
                raise ValueError(
                    "Megatron flex dispatchers require Megatron-Core/Bridge model integration and are not "
                    "wired into this native Metis trainer. Use moe_token_dispatcher_type='alltoall' here, "
                    "or launch a Megatron-Bridge port with DeepEP/HybridEP."
                )
            if self.moe_overlap_expert_parallel_comm or self.moe_delay_wgrad_compute:
                raise ValueError(
                    "moe_overlap_expert_parallel_comm and moe_delay_wgrad_compute require Megatron-Core/Bridge "
                    "EP overlap support; they are not available in the native Metis trainer."
                )
            if self.moe_router_fusion:
                raise ValueError(
                    "moe_router_fusion requires Megatron-Core/Transformer Engine router fusion; it is not "
                    "available in the native Metis trainer."
                )
            if self.moe_expert_parallel_size <= 0:
                raise ValueError("moe_expert_parallel_size must be positive.")
            if self.moe_expert_parallel_size > 1:
                if self.moe_num_experts % self.moe_expert_parallel_size != 0:
                    raise ValueError("moe_num_experts must be divisible by moe_expert_parallel_size.")
                if self.moe_dispatch_mode not in {"grouped", "bucketed"}:
                    raise ValueError("expert-parallel MoE requires grouped or bucketed dispatch.")
                if self.moe_backend not in {"torch_grouped", "torch_grouped_safe", "torch_bmm"}:
                    raise ValueError(
                        "expert-parallel MoE requires a native grouped expert backend: "
                        "torch_grouped, torch_grouped_safe, or torch_bmm."
                    )
            if self.moe_expert_parallel_init_seed < 0:
                raise ValueError("moe_expert_parallel_init_seed cannot be negative.")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
        if self.rope_convention not in {"neox", "legacy_metis14"}:
            raise ValueError("rope_convention must be one of: neox, legacy_metis14.")
        if self.mor_max_depth <= 0:
            raise ValueError("mor_max_depth must be positive.")
        if self.mor_router_hidden_dim <= 0:
            raise ValueError("mor_router_hidden_dim must be positive.")
        if self.mor_router_temperature <= 0:
            raise ValueError("mor_router_temperature must be positive.")
        if self.mor_router_aux_loss_coef < 0:
            raise ValueError("mor_router_aux_loss_coef cannot be negative.")
        if self.mor_router_entropy_coef < 0:
            raise ValueError("mor_router_entropy_coef cannot be negative.")
        if self.mor_router_z_loss_coef < 0:
            raise ValueError("mor_router_z_loss_coef cannot be negative.")
        if self.mor_target_avg_depth < 1.0 or self.mor_target_avg_depth > float(self.mor_max_depth):
            raise ValueError("mor_target_avg_depth must be within [1, mor_max_depth].")
        if self.mor_depth2_capacity_sequences < 0 or self.mor_depth3_capacity_sequences < 0:
            raise ValueError("sequence MoR capacities cannot be negative.")
        if self.mor_block_size <= 0:
            raise ValueError("mor_block_size must be positive.")
        if self.mor_depth2_capacity_blocks < 0 or self.mor_depth3_capacity_blocks < 0:
            raise ValueError("block MoR capacities cannot be negative.")
        if self.block_mor_attention_mode not in {"local_block_refinement"}:
            raise ValueError("block_mor_attention_mode must currently be local_block_refinement.")
        if self.fp8_pad_multiple <= 0:
            raise ValueError("fp8_pad_multiple must be positive.")

    @property
    def padded_vocab_size(self) -> int:
        if self.vocab_size % self.pad_vocab_size_multiple == 0:
            return self.vocab_size
        return self.vocab_size + (
            self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)
        )

    @property
    def uses_mor(self) -> bool:
        return self.mor_enabled and self.training_mode != "static_dense_pretrain"

    @property
    def uses_moe(self) -> bool:
        return self.ffn_type in {"multi_head_latent_moe", "single_latent_moe"}

    @property
    def uses_single_latent_moe(self) -> bool:
        return self.ffn_type == "single_latent_moe"

    @property
    def uses_multi_head_latent_moe(self) -> bool:
        return self.ffn_type == "multi_head_latent_moe"

    @property
    def moe_head_dim(self) -> int:
        return self.d_model // max(self.moe_num_heads, 1)

    @property
    def moe_effective_routed_dim(self) -> int:
        if self.moe_routed_latent_size > 0:
            return self.moe_routed_latent_size
        return self.moe_head_dim

    @property
    def moe_expert_param_factor(self) -> int:
        return 3 if self.moe_activation == "swiglu" else 2

    def nvfp4_surface_precision(self, surface: str) -> str:
        if self.low_precision_mode != "nvfp4":
            return "nvfp4"
        fields = {
            "qkv": (self.nvfp4_qkv_precision, self.nvfp4_keep_qkv_bf16),
            "latent_moe_projection": (
                self.nvfp4_latent_moe_projection_precision,
                self.nvfp4_keep_latent_moe_projections_bf16,
            ),
            "lm_head": (self.nvfp4_lm_head_precision, self.nvfp4_keep_lm_head_bf16),
        }
        if surface not in fields:
            raise ValueError(f"Unknown NVFP4 precision surface: {surface}")
        precision, legacy_keep_bf16 = fields[surface]
        if precision == "legacy":
            return "bf16" if legacy_keep_bf16 else "nvfp4"
        return precision

    def nvfp4_expert_precision_for_layer(self, layer_idx: int | None) -> str:
        if self.low_precision_mode != "nvfp4":
            return "nvfp4"
        if self.nvfp4_final_expert_layers <= 0 or layer_idx is None:
            return "nvfp4"
        first_high_precision_layer = max(0, self.n_layer - self.nvfp4_final_expert_layers)
        if layer_idx >= first_high_precision_layer:
            return self.nvfp4_final_expert_precision
        return "nvfp4"

    def expert_precision_for_layer(self, layer_idx: int | None) -> str:
        if self.low_precision_mode == "fp8":
            return self.fp8_expert_precision
        if self.low_precision_mode == "nvfp4":
            return self.nvfp4_expert_precision_for_layer(layer_idx)
        return "bf16"

    @property
    def nvfp4_uses_mxfp8(self) -> bool:
        if self.low_precision_mode != "nvfp4":
            return False
        return any(
            self.nvfp4_surface_precision(surface) == "mxfp8"
            for surface in ("qkv", "latent_moe_projection", "lm_head")
        ) or self.nvfp4_final_expert_precision == "mxfp8"

    @property
    def nvfp4_uses_fp8_block(self) -> bool:
        if self.low_precision_mode != "nvfp4":
            return False
        return any(
            self.nvfp4_surface_precision(surface) == "fp8_block"
            for surface in ("qkv", "latent_moe_projection", "lm_head")
        ) or self.nvfp4_final_expert_precision == "fp8_block"

    @property
    def uses_dynamic_token_mor(self) -> bool:
        return self.uses_mor and self.training_mode == "dynamic_token_mor"

    @property
    def uses_static_sequence_mor(self) -> bool:
        return self.uses_mor and self.training_mode == "static_sequence_mor"

    @property
    def uses_static_block_mor(self) -> bool:
        return self.uses_mor and self.training_mode == "static_block_mor"

    @property
    def effective_layer_count(self) -> int:
        if not self.uses_mor:
            return self.n_layer
        return self.n_layer * self.mor_max_depth

    @property
    def target_effective_layer_count(self) -> float:
        if not self.uses_mor:
            return float(self.n_layer)
        return self.n_layer * self.mor_target_avg_depth

    @property
    def attn_cfg(self) -> dict[str, Any]:
        return {
            "causal": True,
            "head_dim": self.head_dim,
            "num_heads": self.n_heads,
            "num_heads_kv": self.n_kv_heads,
            "attention_bias": self.attention_bias,
            "dropout": self.attention_dropout,
            "rope_theta": self.rope_theta,
            "backend": self.attention_backend,
            "native_gqa_attention": self.native_gqa_attention,
            "te_dot_product_attention": self.te_dot_product_attention,
        }

    def estimate_params(self) -> int:
        audit = self.param_application_audit()
        embed_params = int(audit["embedding_params"])
        attn_block = int(audit["attention_apps_per_layer"])
        if self.uses_moe:
            routed_expert = int(audit["expert_param_apps_per_assignment"])
            shared_expert = int(audit["shared_expert_param_apps_per_assignment"])
            expert_params = (self.moe_num_experts * routed_expert) + (self.moe_shared_experts * shared_expert)
            router_params = int(audit["router_projection_and_match_apps_per_layer"])
            latent_projection_params = int(audit["latent_projection_apps_per_layer"])
            mlp_block = expert_params + router_params + latent_projection_params
        else:
            mlp_block = 3 * self.d_model * self.intermediate_size
        norm_modules = (2 * self.n_layer) + 1 + (1 if self.uses_mor else 0)
        norm_params = norm_modules * self.d_model
        router_params = 0
        if self.uses_mor:
            router_params += (
                self.d_model * self.mor_router_hidden_dim
                + self.mor_router_hidden_dim
                + self.mor_router_hidden_dim * self.mor_max_depth
                + self.mor_max_depth
            )
        total = embed_params + (self.n_layer * (attn_block + mlp_block)) + norm_params + router_params
        return int(total)

    def estimate_active_params(self, mean_depth: float = 1.0) -> int:
        audit = self.param_application_audit()
        embed_params = int(audit["embedding_params"])
        attn_block = int(audit["attention_apps_per_layer"])
        if self.uses_moe:
            mlp_block = (
                int(audit["routed_expert_param_apps_per_layer"])
                + int(audit["shared_expert_param_apps_per_layer"])
                + int(audit["router_projection_and_match_apps_per_layer"])
                + int(audit["latent_projection_apps_per_layer"])
            )
        else:
            mlp_block = 3 * self.d_model * self.intermediate_size
        norm_modules = (2 * self.n_layer) + 1 + (1 if self.uses_mor else 0)
        norm_params = norm_modules * self.d_model
        mor_router_params = 0
        if self.uses_mor:
            mor_router_params = (
                self.d_model * self.mor_router_hidden_dim
                + self.mor_router_hidden_dim
                + self.mor_router_hidden_dim * self.mor_max_depth
                + self.mor_max_depth
            )
        active_block_params = float(self.n_layer) * (attn_block + mlp_block) * max(1.0, mean_depth)
        return int(embed_params + active_block_params + norm_params + mor_router_params)

    def param_application_audit(self) -> dict[str, int | str]:
        embed = self.padded_vocab_size * self.d_model
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        attn = self.d_model * (q_dim + 2 * kv_dim) + q_dim * self.d_model
        if not self.uses_moe:
            dense_mlp = 3 * self.d_model * self.intermediate_size
            per_layer = attn + dense_mlp
            total = embed + self.n_layer * per_layer
            return {
                "ffn_type": self.ffn_type,
                "d_model": self.d_model,
                "attention_apps_per_layer": attn,
                "dense_mlp_param_apps_per_layer": dense_mlp,
                "embedding_params": embed,
                "rough_total_param_apps_per_token": total,
                "estimated_train_flops_per_token": 6 * total,
            }

        expert_hidden = self.moe_expert_intermediate_size
        routed_dim = self.moe_effective_routed_dim
        expert_assignment = self.moe_expert_param_factor * routed_dim * expert_hidden
        shared_input_dim = self.d_model if self.uses_single_latent_moe else self.moe_head_dim
        shared_assignment = self.moe_expert_param_factor * shared_input_dim * expert_hidden
        routed = self.moe_num_heads * min(self.moe_num_experts, self.moe_top_k) * expert_assignment
        shared = self.moe_num_heads * self.moe_shared_experts * shared_assignment
        if self.uses_single_latent_moe:
            latent_projection = 2 * self.d_model * routed_dim
            router_input_dim = (
                self.d_model
                if self.moe_single_latent_router_input == "hidden"
                else routed_dim
            )
            router_projection = router_input_dim * self.moe_num_experts + self.moe_num_experts
            routing_units = self.moe_top_k
        else:
            latent_projection = (
                0
                if routed_dim == self.moe_head_dim
                else self.moe_num_heads * (2 * self.moe_head_dim * routed_dim)
            )
            router_projection = self.moe_num_heads * (
                self.moe_head_dim * self.moe_router_latent_size
                + self.moe_num_experts * self.moe_router_latent_size
                + self.moe_num_experts
            )
            routing_units = self.moe_num_heads * self.moe_top_k
        per_layer = attn + routed + shared + latent_projection + router_projection
        total = embed + self.n_layer * per_layer
        expert_parallel_size = max(1, int(self.moe_expert_parallel_size))
        routed_expert_params_per_rank = (
            (self.moe_num_heads * self.moe_num_experts * expert_assignment) // expert_parallel_size
        )
        per_rank_per_layer = attn + routed_expert_params_per_rank + shared + latent_projection + router_projection
        params_per_expert_parallel_rank = embed + self.n_layer * per_rank_per_layer
        return {
            "ffn_type": self.ffn_type,
            "d_model": self.d_model,
            "latent_dim": routed_dim,
            "num_experts": self.moe_num_experts,
            "expert_parallel_size": expert_parallel_size,
            "local_experts_per_rank": self.moe_num_experts // expert_parallel_size,
            "top_k": self.moe_top_k,
            "routing_units_per_token": routing_units,
            "expert_hidden": expert_hidden,
            "moe_activation": self.moe_activation,
            "moe_router_override": self.moe_router_override,
            "moe_single_latent_router_input": self.moe_single_latent_router_input,
            "moe_memory_efficient_permutation": self.moe_memory_efficient_permutation,
            "moe_token_dispatcher_type": self.moe_token_dispatcher_type,
            "moe_flex_dispatcher_backend": self.moe_flex_dispatcher_backend,
            "moe_overlap_expert_parallel_comm": self.moe_overlap_expert_parallel_comm,
            "moe_delay_wgrad_compute": self.moe_delay_wgrad_compute,
            "moe_router_fusion": self.moe_router_fusion,
            "moe_permute_fusion": self.moe_permute_fusion,
            "expert_param_apps_per_assignment": expert_assignment,
            "routed_expert_params_per_rank": routed_expert_params_per_rank,
            "shared_expert_param_apps_per_assignment": shared_assignment,
            "routed_expert_param_apps_per_layer": routed,
            "shared_expert_param_apps_per_layer": shared,
            "latent_projection_apps_per_layer": latent_projection,
            "router_projection_and_match_apps_per_layer": router_projection,
            "attention_apps_per_layer": attn,
            "embedding_params": embed,
            "rough_total_param_apps_per_token": total,
            "estimated_params_per_expert_parallel_rank": params_per_expert_parallel_rank,
            "estimated_train_flops_per_token": 6 * total,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimated_params"] = self.estimate_params()
        payload["estimated_active_params"] = self.estimate_active_params(self.mor_target_avg_depth if self.uses_mor else 1.0)
        payload["param_application_audit"] = self.param_application_audit()
        payload["attn_cfg"] = self.attn_cfg
        payload["effective_layer_count"] = self.effective_layer_count
        payload["target_effective_layer_count"] = self.target_effective_layer_count
        return payload
