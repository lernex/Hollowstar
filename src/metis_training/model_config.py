from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


MANIFEST_SCHEMA = "metis.model-family/v1"
_PLACEMENTS = {"replicated", "expert_sharded", "sparse_table"}
_TABLE_MODES = {"replicated", "row_sharded"}
_FFN_MODES = {"moe", "dense"}
_EXPERT_EXECUTIONS = {"loop", "grouped", "grouped_gemm"}
_PATHWAY_MODES = {"per_pass", "frozen"}
# Families whose manifests are research artifacts rather than release contracts.
# Production geometry, context, and pass locks are relaxed for these; every
# structural invariant that the executable model depends on still applies.
_RELAXED_FAMILIES = frozenset({"tiny", "ablation"})
_FAMILIES = frozenset({"praxis", "logos", "tiny", "ablation"})


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _tuple_int(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _tuple_float(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _tuple_str(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class PrecisionConfig:
    """Numerical policy declared by the model manifest.

    Parameters stay in BF16 for ordinary PyTorch execution. The training
    optimizer is required to own FP32 masters and FP32 optimizer states.
    FP8 is an execution policy for eligible GEMMs, not a storage dtype.
    """

    backend: str = "auto"
    fp8_format: str = "hybrid_e4m3_e5m2"
    fp8_scaling: str = "delayed"
    parameter_dtype: str = "bfloat16"
    activation_dtype: str = "bfloat16"
    master_weight_dtype: str = "float32"
    optimizer_state_dtype: str = "float32"
    router_dtype: str = "float32"
    reduction_dtype: str = "float32"
    fp8_roles: tuple[str, ...] = (
        "mamba_in_projection",
        "mamba_out_projection",
        "attention_qkv_projection",
        "attention_out_projection",
        "expert_gate_up_projection",
        "expert_down_projection",
        "latent_down_projection",
        "latent_up_projection",
        "memory_state_write_projection",
        "memory_query_projection",
        "memory_key_projection",
        "memory_value_projection",
        "memory_output_projection",
        "memory_route_projection",
        "mhc_controller",
        "ngram_projection",
        "lm_head",
    )
    bf16_roles: tuple[str, ...] = (
        "embedding",
        "ngram_table",
        "residual_stream",
        "mamba_state",
        "normalization",
        "gate",
        "collective",
        # The per-pass attention LoRA contracts over attention_pass_lora_rank,
        # which is 8 in every family. Transformer Engine cannot execute an FP8
        # GEMM that narrow, and at rank 8 against d_model there is nothing to
        # win by trying: it is a rounding error in both FLOPs and parameters.
        "attention_pass_lora_down",
        "attention_pass_lora_up",
        # memory_metadata_write contracts over route_feature_dim + 4. The +4 is
        # metadata carried beside the route features, and it puts the width 4
        # past a multiple of 16 for every power-of-two route width, so this role
        # cannot be FP8 while that +4 is there.
        "memory_metadata_write_projection",
    )
    fp32_roles: tuple[str, ...] = (
        "master_weight",
        "optimizer_state",
        "router_logits",
        "sinkhorn",
        "memory_attention_logits",
        "loss_accumulation",
        "mamba_sensitive",
    )
    # Wire format for the expert-parallel dispatch/combine all-to-all.  This is
    # a collective payload policy, not a GEMM role, so it stays outside
    # ``fp8_roles`` and outside the sealed precision-role plan.
    #   bfloat16      -- both directions on the wire in BF16
    #   fp8_dispatch  -- FP8 token dispatch, BF16 expert-output combine
    #   fp8           -- both directions in FP8
    # Forward payloads use E4M3 and gradients E5M2, per ``fp8_format``.
    expert_collective_wire: str = "bfloat16"
    require_fp8_validation: bool = True
    allow_bf16_fallback: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "PrecisionConfig":
        if not payload:
            return cls()
        cooked = dict(payload)
        for key in ("fp8_roles", "bf16_roles", "fp32_roles"):
            if key in cooked:
                cooked[key] = _tuple_str(cooked[key])
        return cls(**cooked)

    def validate(self) -> None:
        if self.parameter_dtype != "bfloat16":
            raise ValueError("Metis-1.6 production parameters must use bfloat16 storage.")
        if self.activation_dtype != "bfloat16":
            raise ValueError("Metis-1.6 residual activations must use bfloat16.")
        if self.master_weight_dtype != "float32" or self.optimizer_state_dtype != "float32":
            raise ValueError("Metis-1.6 requires FP32 master weights and optimizer states.")
        if self.router_dtype != "float32" or self.reduction_dtype != "float32":
            raise ValueError("Router logits and sensitive reductions must remain FP32.")
        if self.backend not in {"auto", "transformer_engine", "torch", "bf16"}:
            raise ValueError("precision.backend must be auto, transformer_engine, torch, or bf16.")
        if self.fp8_format not in {
            "e4m3fn",
            "e4m3fnuz",
            "hybrid_e4m3_e5m2",
        }:
            raise ValueError(
                "precision.fp8_format must be e4m3fn, e4m3fnuz, or "
                "hybrid_e4m3_e5m2."
            )
        if self.fp8_scaling not in {"delayed", "current"}:
            raise ValueError("precision.fp8_scaling must be delayed or current.")
        if self.expert_collective_wire not in {"bfloat16", "fp8_dispatch", "fp8"}:
            raise ValueError(
                "precision.expert_collective_wire must be bfloat16, "
                "fp8_dispatch, or fp8."
            )
        if (
            self.expert_collective_wire != "bfloat16"
            and self.fp8_format != "hybrid_e4m3_e5m2"
        ):
            raise ValueError(
                "An FP8 expert collective wire needs hybrid_e4m3_e5m2 so that "
                "forward payloads use E4M3 and gradients use E5M2."
            )


@dataclass(frozen=True)
class NGramMemoryConfig:
    orders: tuple[int, ...] = (2, 3)
    tables_per_order: int = 8
    value_dim: int = 64
    slots_by_order: dict[int, tuple[int, ...]] = field(default_factory=dict)
    hash_seeds: tuple[int, ...] = (
        0x9E3779B1,
        0x85EBCA77,
        0xC2B2AE3D,
        0x27D4EB2F,
        0x165667B1,
        0xD3A2646D,
        0xFD7046C5,
        0xB55A4F09,
    )
    table_mode: str = "replicated"
    sparse_gradients: bool = True
    optimizer: str = "sparse_adam"
    learning_rate_scale: float = 2.0
    weight_decay: float = 0.0
    injection_layers: tuple[int, ...] = (2,)
    canonicalization: str = "tokenizer_sidecar_required"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "NGramMemoryConfig":
        if not payload:
            return cls()
        cooked = dict(payload)
        if "orders" in cooked:
            cooked["orders"] = _tuple_int(cooked["orders"])
        if "hash_seeds" in cooked:
            cooked["hash_seeds"] = _tuple_int(cooked["hash_seeds"])
        if "injection_layers" in cooked:
            cooked["injection_layers"] = _tuple_int(cooked["injection_layers"])
        slots = cooked.get("slots_by_order", {})
        cooked["slots_by_order"] = {
            int(order): _tuple_int(values) for order, values in slots.items()
        }
        return cls(**cooked)

    @property
    def retrieved_rows_per_token(self) -> int:
        return len(self.orders) * self.tables_per_order

    @property
    def concatenated_dim(self) -> int:
        return self.retrieved_rows_per_token * self.value_dim

    @property
    def parameter_count(self) -> int:
        return sum(sum(self.slots_by_order[order]) for order in self.orders) * self.value_dim

    def validate(self) -> None:
        if self.orders != (2, 3):
            raise ValueError("Metis-1.6 conditional memory is locked to suffix orders (2, 3).")
        if self.tables_per_order != 8:
            raise ValueError("Metis-1.6 requires eight independent hash tables per N-gram order.")
        if self.value_dim != 64:
            raise ValueError("Metis-1.6 N-gram table rows are locked to width 64.")
        if len(self.hash_seeds) != self.tables_per_order:
            raise ValueError("hash_seeds must contain one seed per hash head.")
        if self.table_mode not in _TABLE_MODES:
            raise ValueError("ngram_memory.table_mode must be replicated or row_sharded.")
        if self.optimizer != "sparse_adam":
            raise ValueError("N-gram tables require a sparse Adam-style optimizer.")
        if self.weight_decay != 0.0:
            raise ValueError("N-gram sparse tables must not use weight decay.")
        if self.canonicalization != "tokenizer_sidecar_required":
            raise ValueError(
                "Metis-1.6 production N-gram memory requires the tokenizer "
                "canonical-ID sidecar."
            )
        for order in self.orders:
            slots = self.slots_by_order.get(order)
            if slots is None or len(slots) != self.tables_per_order:
                raise ValueError(f"slots_by_order[{order}] must contain eight prime-sized tables.")
            if any(slot <= 2 for slot in slots):
                raise ValueError("N-gram table slot counts must be greater than two.")
            if any(not _is_prime(slot) for slot in slots):
                raise ValueError("Every N-gram hash-table slot count must be prime.")

    def validate_relaxed(self) -> None:
        """Validate structure without the production table-size constants.

        Research families scale the table geometry with the model, so
        ``tables_per_order``, ``value_dim``, and the canonicalization policy are
        free.  Everything the lookup kernel actually depends on -- suffix
        orders, one seed per hash head, one prime slot count per head -- is
        still enforced, because those are correctness constraints rather than
        release contracts.
        """

        if self.orders != (2, 3):
            raise ValueError("Metis-1.6 conditional memory is locked to suffix orders (2, 3).")
        if self.tables_per_order <= 0:
            raise ValueError("tables_per_order must be positive.")
        if self.value_dim <= 0:
            raise ValueError("N-gram table rows must have positive width.")
        if len(self.hash_seeds) != self.tables_per_order:
            raise ValueError("hash_seeds must contain one seed per hash head.")
        if self.table_mode not in _TABLE_MODES:
            raise ValueError("ngram_memory.table_mode must be replicated or row_sharded.")
        if self.optimizer != "sparse_adam":
            raise ValueError("N-gram tables require a sparse Adam-style optimizer.")
        if self.weight_decay != 0.0:
            raise ValueError("N-gram sparse tables must not use weight decay.")
        for order in self.orders:
            slots = self.slots_by_order.get(order)
            if slots is None or len(slots) != self.tables_per_order:
                raise ValueError(
                    f"slots_by_order[{order}] must contain one prime slot count per hash head."
                )
            if any(slot <= 2 for slot in slots):
                raise ValueError("N-gram table slot counts must be greater than two.")
            if any(not _is_prime(slot) for slot in slots):
                raise ValueError("Every N-gram hash-table slot count must be prime.")


@dataclass(frozen=True)
class AutotuneGates:
    max_hbm_fraction: float = 0.90
    max_fp8_loss_relative_error: float = 0.02
    max_ngram_layout_loss_relative_error: float = 0.001
    max_update_to_weight_ratio: float = 0.01
    max_grad_norm: float = 100.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "AutotuneGates":
        return cls(**dict(payload or {}))

    def validate(self) -> None:
        if not 0.0 < self.max_hbm_fraction < 1.0:
            raise ValueError("autotune.gates.max_hbm_fraction must be in (0, 1).")
        if self.max_fp8_loss_relative_error < 0.0:
            raise ValueError("max_fp8_loss_relative_error cannot be negative.")
        if self.max_ngram_layout_loss_relative_error < 0.0:
            raise ValueError("max_ngram_layout_loss_relative_error cannot be negative.")
        if self.max_update_to_weight_ratio <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("Autotune optimizer safety gates must be positive.")


@dataclass(frozen=True)
class GlobalTokenBatchBounds:
    minimum: int
    maximum: int
    target: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GlobalTokenBatchBounds":
        cooked = dict(payload)
        if "min" in cooked:
            cooked["minimum"] = cooked.pop("min")
        if "max" in cooked:
            cooked["maximum"] = cooked.pop("max")
        return cls(**{key: int(value) for key, value in cooked.items()})

    def validate(self) -> None:
        if not 0 < self.minimum <= self.target <= self.maximum:
            raise ValueError("global_token_batch must satisfy 0 < min <= target <= max.")


@dataclass(frozen=True)
class AutotuneConfig:
    status: str = "bounded_canary_candidates_not_validated_optima"
    micro_batch_sizes: tuple[int, ...] = (1,)
    grad_accum_steps: tuple[int, ...] = (1,)
    global_token_batch: GlobalTokenBatchBounds = field(
        default_factory=lambda: GlobalTokenBatchBounds(1, 1, 1)
    )
    learning_rates: tuple[float, ...] = (1.0e-4,)
    preferred_learning_rate: float = 1.0e-4
    compile_modes: tuple[str, ...] = ("default", "reduce-overhead", "none")
    precision_profiles: tuple[str, ...] = ("fp8", "bf16")
    dispatch_overlap: tuple[str, ...] = ("on", "off")
    ngram_table_modes: tuple[str, ...] = ("replicated", "row_sharded")
    gates: AutotuneGates = field(default_factory=AutotuneGates)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "AutotuneConfig":
        if not payload:
            return cls()
        bounds = dict(payload.get("bounds", payload))
        gates = AutotuneGates.from_mapping(payload.get("gates"))
        return cls(
            status=str(payload.get("status", "bounded_canary_candidates_not_validated_optima")),
            micro_batch_sizes=_tuple_int(bounds["micro_batch_sizes"]),
            grad_accum_steps=_tuple_int(bounds["grad_accum_steps"]),
            global_token_batch=GlobalTokenBatchBounds.from_mapping(bounds["global_token_batch"]),
            learning_rates=_tuple_float(bounds["learning_rates"]),
            preferred_learning_rate=float(
                bounds.get("preferred_learning_rate", bounds["learning_rates"][0])
            ),
            compile_modes=_tuple_str(bounds["compile_modes"]),
            precision_profiles=_tuple_str(bounds["precision_profiles"]),
            dispatch_overlap=_tuple_str(bounds["dispatch_overlap"]),
            ngram_table_modes=_tuple_str(bounds["ngram_table_modes"]),
            gates=gates,
        )

    def validate(self) -> None:
        if self.status != "bounded_canary_candidates_not_validated_optima":
            raise ValueError("Autotune candidates must be marked as unvalidated bounded canary candidates.")
        if tuple(sorted(self.micro_batch_sizes, reverse=True)) != self.micro_batch_sizes:
            raise ValueError("autotune micro_batch_sizes must be descending.")
        if not self.micro_batch_sizes or any(value <= 0 for value in self.micro_batch_sizes):
            raise ValueError("autotune micro_batch_sizes must be non-empty and positive.")
        if not self.grad_accum_steps or any(value <= 0 for value in self.grad_accum_steps):
            raise ValueError("autotune grad_accum_steps must be non-empty and positive.")
        if not self.learning_rates or any(value <= 0 for value in self.learning_rates):
            raise ValueError("autotune learning_rates must be non-empty and positive.")
        if self.preferred_learning_rate not in self.learning_rates:
            raise ValueError("preferred_learning_rate must be one of the bounded learning_rates.")
        if not set(self.precision_profiles) <= {"fp8", "bf16"}:
            raise ValueError("autotune precision_profiles may contain only fp8 and bf16.")
        if not set(self.dispatch_overlap) <= {"on", "off"}:
            raise ValueError("autotune dispatch_overlap may contain only on and off.")
        if (
            not self.ngram_table_modes
            or not set(self.ngram_table_modes) <= _TABLE_MODES
        ):
            raise ValueError(
                "autotune ngram_table_modes may contain only replicated and row_sharded."
            )
        self.global_token_batch.validate()
        self.gates.validate()


@dataclass(frozen=True)
class ParameterAudit:
    embedding: int
    mixers: int
    mhc: int
    latent_projections: int
    routed_experts: int
    shared_experts: int
    expert_routers: int
    depth_memory: int
    continuation: int
    ngram_tables: int
    ngram_fusion: int
    final_norm: int
    stored_total: int
    active_per_pass_min: int
    active_per_pass_mean: int
    active_per_pass_max: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Metis16Config:
    schema: str = MANIFEST_SCHEMA
    family: str = "praxis"
    name: str = "Metis-1.6-Praxis"
    model_type: str = "metis16_more"
    vocab_size: int = 65_536
    sequence_length: int = 4_096
    final_context_length: int = 131_072
    context_extension_train_length: int = 163_840
    d_model: int = 2_048
    n_layers: int = 12
    attention_indices: tuple[int, ...] = (4, 8)
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 64
    no_position_embeddings: bool = True
    attention_pass_lora_rank: int = 8
    attention_backend: str = "varlen_fused_required"
    mamba_expand: int = 2
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_head_dim: int = 64
    mamba_ngroups: int = 8
    mamba_chunk_size: int = 256
    mamba_backend: str = "fused_required"
    n_streams: int = 4
    mhc_pass_embedding_dim: int = 64
    mhc_sinkhorn_iterations: int = 8
    mhc_backend: str = "fused_required"
    latent_dim: int = 1_024
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    expert_intermediate_dim: int = 512
    min_routed_k: int = 1
    max_routed_k: int = 8
    target_mean_routed_k: float = 4.0
    # Feed-forward sublayer family.  ``moe`` is the production Metis-1.6 path.
    # ``dense`` replaces the routed/shared expert mixture with a single SwiGLU
    # block and is only reachable from the ``ablation`` family, where the paper
    # needs dense controls that are otherwise architecturally identical.
    ffn_mode: str = "moe"
    dense_ffn_intermediate_dim: int = 0
    # ``loop`` issues one GEMM pair per local expert, which is what production
    # expert parallelism wants because each rank owns a handful of experts.
    # ``grouped`` sorts the assignments once and derives segment boundaries with
    # a single host synchronization per layer, but still dispatches per expert.
    # ``grouped_gemm`` contracts the whole bank in one GEMM per projection, so
    # the dispatch count stops scaling with the expert count -- the only viable
    # path when every rank replicates all routed experts.
    expert_execution: str = "loop"
    world_size: int = 128
    expert_parallel_size: int = 128
    expert_replicas: int = 1
    max_passes: int = 5
    target_mean_passes: float = 2.0
    route_feature_dim: int = 256
    memory_dim: int = 256
    memory_heads: int = 4
    memory_gate_init: float = -6.0
    continuation_gate_init: float = 0.0
    router_z_loss_coefficient: float = 1.0e-3
    expert_balance_coefficient: float = 0.0
    expert_balance_bias_update_rate: float = 1.0e-3
    k_budget_coefficient: float = 1.0e-2
    depth_budget_coefficient: float = 1.0e-2
    # Dual step size for the width and depth budget controllers. Zero keeps the
    # fixed-coefficient penalty exactly as it was, which is what production
    # Praxis and Logos run. Any positive value turns the coefficient above into
    # the quadratic term of an augmented Lagrangian and lets a multiplier find
    # the strength the constraint actually needs; see
    # ``metis_training.model.BudgetController``.
    budget_controller_rate: float = 0.0
    # The width budget is evaluated once per MoE layer per pass and the depth
    # budget once per forward, so at equal rates the depth multiplier receives
    # roughly twenty times less integral gain per optimizer step and takes
    # twenty times longer to bind. Measured: width falls 7.35 -> 3.37 in fifty
    # steps while depth climbs 1.83 -> 5.00 in ten. Give depth its own rate.
    depth_budget_controller_rate: float = 0.0
    # Exponential forgetting on the budget multipliers. A pure integrator winds
    # up while the policy it controls is pinned against a limit, and delivers
    # the whole accumulated total the moment the policy moves.
    budget_controller_leak: float = 0.0
    budget_multiplier_limit: float = 1.0e3
    continuation_entropy_coefficient: float = 1.0e-3
    tie_embeddings: bool = True
    # Software-pipeline depth for the expert-parallel dispatch/expert/combine
    # chain.  One reproduces the serial path byte for byte; higher values keep
    # the same bytes on the wire but overlap them with the expert GEMMs.
    moe_dispatch_chunks: int = 1
    lm_head_chunk_size: int = 1_024
    # ``pass`` replays a whole recurrent pass; ``layer`` replays one block at a
    # time.  The FLOP cost is identical -- the same forward is re-executed
    # either way -- but ``layer`` keeps only one block's activations live, which
    # is the difference between fitting and not fitting at 163,840 tokens.
    activation_recompute_policy: str = "pass"
    # Sequence sharding for context extension.  1 leaves every code path byte
    # identical to pretraining; above 1 each rank owns ``sequence / size``
    # contiguous positions and the mixers exchange the state that crosses the
    # boundary.
    context_parallel_size: int = 1
    ngram_memory: NGramMemoryConfig = field(default_factory=NGramMemoryConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    autotune: AutotuneConfig = field(default_factory=AutotuneConfig)
    expected_parameter_audit: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Metis16Config":
        cooked = dict(payload)
        if "architecture" in cooked:
            architecture = dict(cooked.pop("architecture"))
            cooked.update(architecture)
        if "parallelism" in cooked:
            cooked.update(dict(cooked.pop("parallelism")))
        if "topology" in cooked:
            topology = dict(cooked.pop("topology"))
            if "expert_replica_count" in topology:
                topology["expert_replicas"] = topology.pop("expert_replica_count")
            cooked.update(topology)
        if "routing" in cooked:
            cooked.update(dict(cooked.pop("routing")))
        if "depth_memory" in cooked:
            cooked.update(dict(cooked.pop("depth_memory")))
        if "attention_indices" in cooked:
            cooked["attention_indices"] = _tuple_int(cooked["attention_indices"])
        cooked["ngram_memory"] = NGramMemoryConfig.from_mapping(cooked.get("ngram_memory"))
        cooked["precision"] = PrecisionConfig.from_mapping(cooked.get("precision"))
        cooked["autotune"] = AutotuneConfig.from_mapping(cooked.get("autotune"))
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(cooked) - allowed)
        if unknown:
            raise ValueError(f"Unknown Metis-1.6 manifest fields: {unknown}")
        config = cls(**cooked)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Metis16Config":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Model manifest must be a mapping: {path}")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ngram_memory"]["slots_by_order"] = {
            str(order): list(slots) for order, slots in self.ngram_memory.slots_by_order.items()
        }
        return payload

    @property
    def padded_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def n_attention_layers(self) -> int:
        return len(self.attention_indices)

    @property
    def n_mamba_layers(self) -> int:
        return self.n_layers - self.n_attention_layers

    @property
    def memory_slots(self) -> int:
        return self.max_passes * (self.n_attention_layers + 1)

    def validate(self) -> None:
        if self.schema != MANIFEST_SCHEMA:
            raise ValueError(f"Expected schema {MANIFEST_SCHEMA!r}, got {self.schema!r}.")
        if self.family not in _FAMILIES:
            raise ValueError("family must be praxis, logos, tiny, or ablation.")
        if self.ffn_mode not in _FFN_MODES:
            raise ValueError("ffn_mode must be moe or dense.")
        if self.expert_execution not in _EXPERT_EXECUTIONS:
            raise ValueError("expert_execution must be loop, grouped, or grouped_gemm.")
        if self.ffn_mode == "dense" and self.family not in _RELAXED_FAMILIES:
            raise ValueError(
                "A dense feed-forward sublayer is an ablation control; production "
                "Praxis/Logos are locked to the routed expert mixture."
            )
        if self.vocab_size != 65_536 and self.family not in _RELAXED_FAMILIES:
            raise ValueError("Praxis and Logos require the locked 65,536-token vocabulary.")
        if self.sequence_length != 4_096 and self.family not in _RELAXED_FAMILIES:
            raise ValueError("Praxis and Logos pretraining context is locked to 4,096.")
        if self.final_context_length != 131_072 and self.family not in _RELAXED_FAMILIES:
            raise ValueError("Praxis and Logos deployment context is locked to 131,072.")
        if self.context_extension_train_length != 163_840 and self.family not in _RELAXED_FAMILIES:
            raise ValueError("Context-extension training is locked to the 163,840-token overshoot.")
        if self.context_extension_train_length < self.final_context_length:
            raise ValueError("context_extension_train_length cannot be below deployment context.")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if not self.no_position_embeddings:
            raise ValueError("Metis-1.6 attention is locked to NoPE.")
        if self.attention_backend not in {"varlen_fused_required", "auto", "torch_reference"}:
            raise ValueError(
                "attention_backend must be varlen_fused_required, auto, or torch_reference."
            )
        if sorted(set(self.attention_indices)) != list(self.attention_indices):
            raise ValueError("attention_indices must be sorted and unique.")
        if any(index < 0 or index >= self.n_layers for index in self.attention_indices):
            raise ValueError("attention_indices must refer to physical layers.")
        if self.mamba_expand != 2 or self.mamba_d_state != 128:
            if self.family not in _RELAXED_FAMILIES:
                raise ValueError("Production Metis-1.6 uses Mamba-2 expand=2 and d_state=128.")
        if self.mamba_head_dim <= 0 or (self.d_model * self.mamba_expand) % self.mamba_head_dim:
            raise ValueError("Expanded Mamba width must be divisible by mamba_head_dim.")
        if (self.d_model * self.mamba_expand // self.mamba_head_dim) % self.mamba_ngroups:
            raise ValueError("Mamba heads must be divisible by mamba_ngroups.")
        if self.mamba_backend not in {"fused_required", "auto", "torch_reference"}:
            raise ValueError("mamba_backend must be fused_required, auto, or torch_reference.")
        if self.n_streams != 4:
            raise ValueError("Metis-1.6 requires exactly four persistent mHC streams.")
        if self.mhc_sinkhorn_iterations < 2:
            raise ValueError("mHC requires at least two Sinkhorn iterations.")
        if self.mhc_backend not in {"fused_required", "torch_reference"}:
            raise ValueError("mhc_backend must be fused_required or torch_reference.")
        if self.family not in _RELAXED_FAMILIES and self.mhc_backend != "fused_required":
            raise ValueError(
                "Production Praxis/Logos require the fused Triton/ROCm mHC backend."
            )
        if self.latent_dim <= 0 or self.latent_dim > self.d_model:
            raise ValueError("latent_dim must be positive and no wider than d_model.")
        if self.ffn_mode == "dense":
            # A dense control has no expert mixture at all: routed and shared
            # expert counts must be zero so the parameter audit stays honest
            # rather than charging the model for weights it never builds.
            if self.n_routed_experts != 0 or self.n_shared_experts != 0:
                raise ValueError(
                    "ffn_mode=dense requires n_routed_experts and n_shared_experts "
                    "to be zero; the audit must not charge unbuilt experts."
                )
            if self.dense_ffn_intermediate_dim <= 0:
                raise ValueError("ffn_mode=dense requires a positive dense_ffn_intermediate_dim.")
            if self.expert_parallel_size != 1:
                raise ValueError("A dense feed-forward stack cannot use expert parallelism.")
        else:
            if self.n_routed_experts <= 0 or self.n_shared_experts != 1:
                raise ValueError(
                    "Metis-1.6 requires routed experts plus exactly one shared expert."
                )
            if self.n_routed_experts % self.expert_parallel_size:
                raise ValueError(
                    "n_routed_experts must divide evenly across expert_parallel_size."
                )
            if not 1 <= self.min_routed_k <= self.max_routed_k <= self.n_routed_experts:
                raise ValueError("Dynamic routed k bounds are invalid.")
            if not self.min_routed_k <= self.target_mean_routed_k <= self.max_routed_k:
                raise ValueError("target_mean_routed_k must lie inside the routed-k bounds.")
        if self.world_size != self.expert_parallel_size * self.expert_replicas:
            raise ValueError("world_size must equal expert_parallel_size * expert_replicas.")
        if self.expert_balance_coefficient != 0.0 and self.family not in _RELAXED_FAMILIES:
            raise ValueError(
                "Production Metis-1.6 uses aux-loss-free expert bias balancing."
            )
        if self.expert_balance_bias_update_rate <= 0.0:
            raise ValueError("expert_balance_bias_update_rate must be positive.")
        if self.max_passes != 5 and self.family not in _RELAXED_FAMILIES:
            raise ValueError("Production Metis-1.6 is locked to five passes.")
        if not 1.0 <= self.target_mean_passes <= self.max_passes:
            raise ValueError("target_mean_passes must lie in [1, max_passes].")
        if self.memory_dim % self.memory_heads:
            raise ValueError("memory_dim must be divisible by memory_heads.")
        if not 1 <= self.moe_dispatch_chunks <= 8:
            raise ValueError("moe_dispatch_chunks must be between 1 and 8.")
        if self.lm_head_chunk_size <= 0:
            raise ValueError("lm_head_chunk_size must be positive.")
        self._validate_memory_policies()
        if self.family == "ablation":
            # The ablation proxy keeps the production backbone -- four mHC
            # streams, the hybrid mixer stack, N-gram conditional memory -- but
            # scales the table geometry with the model.  Structural invariants
            # (prime slots, one seed per head, matching head counts) still hold.
            self.ngram_memory.validate_relaxed()
        else:
            self.ngram_memory.validate()
        self.precision.validate()
        self.autotune.validate()
        if self.family == "praxis":
            locked = (self.d_model, self.n_layers, self.n_routed_experts, self.expert_intermediate_dim)
            if locked != (2_048, 12, 128, 512):
                raise ValueError("Praxis geometry diverges from the locked Metis-1.6 plan.")
        if self.family == "logos":
            locked = (self.d_model, self.n_layers, self.n_routed_experts, self.expert_intermediate_dim)
            if locked != (2_560, 20, 192, 768):
                raise ValueError("Logos geometry diverges from the locked Metis-1.6 plan.")
        if self.expected_parameter_audit:
            actual = self.logical_parameter_audit().to_dict()
            mismatches = {
                key: (int(expected), int(actual.get(key, -1)))
                for key, expected in self.expected_parameter_audit.items()
                if int(expected) != int(actual.get(key, -1))
            }
            if mismatches:
                raise ValueError(f"Manifest parameter audit is stale: {mismatches}")

    def with_overrides(self, **changes: Any) -> "Metis16Config":
        config = replace(self, **changes)
        config.validate()
        return config

    @classmethod
    def tiny_for_tests(
        cls,
        *,
        table_mode: str = "replicated",
        mamba_backend: str = "torch_reference",
    ) -> "Metis16Config":
        ngram = NGramMemoryConfig(
            slots_by_order={2: (101, 103), 3: (107, 109)},
            tables_per_order=2,
            hash_seeds=(1019, 2027),
            value_dim=8,
            table_mode=table_mode,
            injection_layers=(1,),
        )
        # Tiny deliberately relaxes the production table/head constants.
        object.__setattr__(ngram, "orders", (2, 3))
        autotune = AutotuneConfig(
            micro_batch_sizes=(2, 1),
            grad_accum_steps=(1, 2),
            global_token_batch=GlobalTokenBatchBounds(16, 128, 64),
            learning_rates=(1.0e-3,),
            preferred_learning_rate=1.0e-3,
        )
        config = cls(
            family="tiny",
            name="Metis-1.6-Tiny-Test",
            vocab_size=128,
            sequence_length=16,
            final_context_length=64,
            context_extension_train_length=80,
            d_model=32,
            n_layers=3,
            attention_indices=(1,),
            n_heads=4,
            n_kv_heads=2,
            head_dim=8,
            attention_pass_lora_rank=4,
            attention_backend="torch_reference",
            mamba_d_state=8,
            mamba_d_conv=3,
            mamba_head_dim=8,
            mamba_ngroups=2,
            mamba_chunk_size=8,
            mamba_backend=mamba_backend,
            mhc_pass_embedding_dim=16,
            mhc_backend="torch_reference",
            latent_dim=16,
            n_routed_experts=4,
            expert_parallel_size=1,
            world_size=1,
            expert_intermediate_dim=12,
            min_routed_k=1,
            max_routed_k=3,
            target_mean_routed_k=2.0,
            max_passes=3,
            target_mean_passes=2.0,
            route_feature_dim=16,
            memory_dim=16,
            memory_heads=2,
            ngram_memory=ngram,
            precision=PrecisionConfig(
                backend="bf16",
                require_fp8_validation=False,
            ),
            autotune=autotune,
            activation_recompute_policy="none",
        )
        # Production-only constants in NGramMemoryConfig.validate are relaxed
        # here by validating the full config through the tiny-specific path.
        config._validate_tiny()
        return config

    def _validate_memory_policies(self) -> None:
        """Validate the two knobs that decide whether a step fits in HBM.

        Shared by the production and tiny validation paths so a manifest cannot
        enable context parallelism on one lane and silently ignore it on the
        other.
        """

        if self.activation_recompute_policy not in {"none", "pass", "layer"}:
            raise ValueError(
                "activation_recompute_policy must be none, pass, or layer."
            )
        if self.context_parallel_size < 1:
            raise ValueError("context_parallel_size must be at least 1.")
        if self.context_parallel_size > 1:
            for name, length in (
                ("context_extension_train_length", self.context_extension_train_length),
                ("final_context_length", self.final_context_length),
            ):
                if length % self.context_parallel_size:
                    raise ValueError(
                        f"{name}={length} is not divisible by "
                        f"context_parallel_size={self.context_parallel_size}; "
                        "Metis-1.6 requires equal sequence shards."
                    )
            if self.activation_recompute_policy == "none":
                # Sharding the sequence without recomputing activations is a
                # configuration that only ever arises by accident: it spends the
                # communication and keeps the memory.
                raise ValueError(
                    "context_parallel_size above 1 requires activation "
                    "recompute; use policy 'layer' for context extension."
                )

    def _validate_tiny(self) -> None:
        if self.family != "tiny":
            raise ValueError("_validate_tiny is only valid for tiny configs.")
        self._validate_memory_policies()
        if self.mhc_backend != "torch_reference":
            raise ValueError("Tiny/CPU tests require the torch mHC reference backend.")
        if self.ngram_memory.orders != (2, 3):
            raise ValueError("Tiny tests still require 2/3-gram memory.")
        if len(self.ngram_memory.hash_seeds) != self.ngram_memory.tables_per_order:
            raise ValueError("Tiny N-gram hash heads are inconsistent.")
        for order in self.ngram_memory.orders:
            if len(self.ngram_memory.slots_by_order[order]) != self.ngram_memory.tables_per_order:
                raise ValueError("Tiny N-gram table geometry is inconsistent.")
        self.precision.validate()
        self.autotune.validate()

    @property
    def expert_entropy_normalizer(self) -> float:
        """Denominator that turns router entropy into a 0-1 ratio.

        A dense ablation control has no routed experts, so the natural
        normalizer ``log(n_routed_experts)`` is undefined.  Returning 1.0 makes
        the ratio identically zero there, which is the truthful reading: a
        sublayer with no routing decision has no routing uncertainty.
        """

        import math as _math

        if self.n_routed_experts <= 1:
            return 1.0
        return _math.log(float(self.n_routed_experts))

    def active_parameters_per_pass(self, routed_k: float) -> float:
        """Active parameters for one pass at a given routed-k.

        ``logical_parameter_audit`` reports min, mean and max at the config's
        own ``min``/``target_mean``/``max`` routed-k. A row that fixes k
        elsewhere -- the k=8 MoE control, say -- is neither of those, and
        reporting its mean would understate it by four experts a layer while
        its FLOPs, which are computed from the row's actual k, say otherwise.
        The count is linear in k by construction, so interpolate rather than
        keep a second copy of the expert arithmetic.
        """

        audit = self.logical_parameter_audit()
        span = self.max_routed_k - self.min_routed_k
        if span <= 0:
            return float(audit.active_per_pass_mean)
        slope = (audit.active_per_pass_max - audit.active_per_pass_min) / span
        return float(
            audit.active_per_pass_min + (float(routed_k) - self.min_routed_k) * slope
        )

    def logical_parameter_audit(self) -> ParameterAudit:
        """Return exact logical counts for the modules implemented in model.py."""

        d = self.d_model
        latent = self.latent_dim
        expert_hidden = self.expert_intermediate_dim
        n_mamba = self.n_mamba_layers
        n_attn = self.n_attention_layers

        embedding = self.vocab_size * d
        mamba_inner = d * self.mamba_expand
        mamba_heads = mamba_inner // self.mamba_head_dim
        mamba_in_width = (2 * mamba_inner) + (2 * self.mamba_ngroups * self.mamba_d_state) + mamba_heads
        mamba_per_layer = (
            d * mamba_in_width
            + (mamba_inner + 2 * self.mamba_ngroups * self.mamba_d_state) * self.mamba_d_conv
            + (mamba_inner + 2 * self.mamba_ngroups * self.mamba_d_state)
            + mamba_heads * 3
            + mamba_inner
            + mamba_inner * d
        )
        qkv_width = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        attention_per_layer = (
            d * qkv_width
            + self.attention_pass_lora_rank * (d + qkv_width)
            + d * d
            + self.max_passes
        )
        mixers = n_mamba * mamba_per_layer + n_attn * attention_per_layer

        # Two mHC connections per block. Each owns read logits, base matrix,
        # write logits, pass-controller projection/bias, and a normalization.
        mhc_per_connection = (
            self.n_streams
            + self.n_streams * self.n_streams
            + self.n_streams
            + self.mhc_pass_embedding_dim * (self.n_streams * self.n_streams + 2 * self.n_streams)
            + (self.n_streams * self.n_streams + 2 * self.n_streams)
            + d
        )
        pass_embeddings = self.max_passes * self.mhc_pass_embedding_dim
        stream_embeddings = self.n_streams * d
        mhc = self.n_layers * 2 * mhc_per_connection + pass_embeddings + stream_embeddings

        if self.ffn_mode == "dense":
            # The dense control keeps the latent bottleneck so the sublayer's
            # shape is comparable to the mixture it replaces, and spends its
            # whole feed-forward budget on one SwiGLU block per layer.
            latent_per_layer = d * latent + latent * d
            latent_projections = self.n_layers * latent_per_layer
            routed_experts = 0
            shared_experts = self.n_layers * 3 * latent * self.dense_ffn_intermediate_dim
            expert_routers = 0
        else:
            latent_per_layer = d * latent + latent * d
            latent_projections = self.n_layers * latent_per_layer
            expert_per = 3 * latent * expert_hidden
            routed_experts = self.n_layers * self.n_routed_experts * expert_per
            shared_experts = self.n_layers * self.n_shared_experts * expert_per
            route_input = latent + self.route_feature_dim
            router_per_layer = (
                route_input * self.n_routed_experts
                + self.n_routed_experts
                + route_input * (self.max_routed_k - self.min_routed_k + 1)
                + (self.max_routed_k - self.min_routed_k + 1)
                + self.n_routed_experts * self.route_feature_dim
            )
            expert_routers = self.n_layers * router_per_layer

        # Global recurrent-memory projections and typed metadata.
        memory_metadata_width = self.route_feature_dim + 4
        depth_memory = (
            d * self.memory_dim
            + self.memory_dim
            + memory_metadata_width * self.memory_dim
            + self.memory_dim
            + self.max_passes * self.memory_dim
            + (self.n_attention_layers + 1) * self.memory_dim
            + d * self.memory_dim
            + self.memory_dim
            + self.memory_dim * self.memory_dim
            + self.memory_dim
            + self.memory_dim * self.memory_dim
            + self.memory_dim
            + self.memory_dim * d
            + d
            + (3 * d + self.route_feature_dim) * self.route_feature_dim
            + self.route_feature_dim
            + self.n_streams * d
            + self.n_streams
        )
        continuation_input = d * 3 + self.route_feature_dim
        continuation = (
            continuation_input * self.route_feature_dim
            + self.route_feature_dim
            + self.route_feature_dim
            + 1
        )

        ngram_tables = self.ngram_memory.parameter_count
        ngram_fusion = (
            self.ngram_memory.concatenated_dim * d
            + d
            + len(self.ngram_memory.injection_layers)
            * self.max_passes
            * self.n_streams
            * (d + 1)
        )
        final_norm = d
        stored_total = (
            embedding
            + mixers
            + mhc
            + latent_projections
            + routed_experts
            + shared_experts
            + expert_routers
            + depth_memory
            + continuation
            + ngram_tables
            + ngram_fusion
            + final_norm
        )

        common_active = (
            mixers
            + mhc
            + latent_projections
            + shared_experts
            + expert_routers
            + depth_memory
            + continuation
            + ngram_fusion
            + final_norm
        )
        # A dense sublayer activates all of its feed-forward weight on every
        # token, so its routed-per-k slope is zero and min/mean/max collapse to
        # the same value.  That is the whole point of the dense control.
        routed_per_k = 0 if self.ffn_mode == "dense" else self.n_layers * expert_per
        # Only retrieved table rows are active; count them once per sequence
        # token because they are cached across recurrent passes.
        retrieved = self.ngram_memory.concatenated_dim
        return ParameterAudit(
            embedding=embedding,
            mixers=mixers,
            mhc=mhc,
            latent_projections=latent_projections,
            routed_experts=routed_experts,
            shared_experts=shared_experts,
            expert_routers=expert_routers,
            depth_memory=depth_memory,
            continuation=continuation,
            ngram_tables=ngram_tables,
            ngram_fusion=ngram_fusion,
            final_norm=final_norm,
            stored_total=stored_total,
            active_per_pass_min=int(common_active + self.min_routed_k * routed_per_k + retrieved),
            active_per_pass_mean=int(common_active + self.target_mean_routed_k * routed_per_k + retrieved),
            active_per_pass_max=int(common_active + self.max_routed_k * routed_per_k + retrieved),
        )


def default_manifest_path(family: str) -> Path:
    normalized = family.strip().lower()
    if normalized not in {"praxis", "logos"}:
        raise ValueError("family must be praxis or logos.")
    return Path(__file__).resolve().parents[2] / "configs" / "metis16" / f"{normalized}.yaml"


def load_family_config(
    source: str | Path | Mapping[str, Any] | None = None,
    *,
    family: str | None = None,
    materialize_tables: bool = True,
) -> Metis16Config:
    """Load and validate an executable Praxis or Logos model manifest.

    ``materialize_tables`` is accepted as a stable trainer API flag. Loading a
    config never allocates table tensors; allocation occurs in the model
    constructor, so the flag intentionally has no side effect.
    """

    del materialize_tables
    if source is None:
        if family is None:
            raise ValueError("Provide either source or family.")
        source = default_manifest_path(family)
    if isinstance(source, Mapping):
        config = Metis16Config.from_mapping(source)
    else:
        path = Path(source)
        if path.exists():
            config = Metis16Config.from_yaml(path)
        elif family is None and str(source).lower() in {"praxis", "logos"}:
            config = Metis16Config.from_yaml(default_manifest_path(str(source)))
        else:
            raise FileNotFoundError(path)
    if family is not None and config.family != family.lower():
        raise ValueError(f"Manifest family {config.family!r} does not match requested {family!r}.")
    return config


__all__ = [
    "AutotuneConfig",
    "AutotuneGates",
    "GlobalTokenBatchBounds",
    "MANIFEST_SCHEMA",
    "Metis16Config",
    "NGramMemoryConfig",
    "ParameterAudit",
    "PrecisionConfig",
    "default_manifest_path",
    "load_family_config",
]
