"""The thirteen-row MoRE ablation ladder as executable specifications.

Every row is the same proxy geometry on the same backbone -- hybrid Mamba-2 and
attention mixers, four mHC streams, N-gram conditional memory -- and differs
only in the routing axes under test.  Holding the backbone constant makes every
baseline stronger than its standard counterpart, which is the conservative
direction for the MoRE claim; see ``docs/papers/more/main.tex`` section 6.1.

Two invariants are enforced here rather than trusted:

* **Identical global batch.** Every row consumes exactly
  ``GLOBAL_BATCH_SEQUENCES`` sequences per optimizer step regardless of how many
  APUs it was given, so all thirteen models see byte-identical token sets in
  identical order.  Data order is therefore not a confound for anyone.
* **Iso-FLOP by construction.** Rows 1 and 5-13 execute the same FLOPs per
  token.  Rows 2, 3, and 4 are deliberate off-frontier reference points and are
  labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from metis_training.model import CurriculumState
from metis_training.model_config import (
    AutotuneConfig,
    AutotuneGates,
    GlobalTokenBatchBounds,
    Metis16Config,
    NGramMemoryConfig,
    PrecisionConfig,
)


# 448 = 2^6 * 7, so it divides every APU count in the campaign (16, 28, 32, 64)
# and the per-rank sequence count is always an integer.  At 4096 tokens that is
# a 1,835,008-token optimizer step -- large enough to hide the gradient
# all-reduce behind compute (see ablation_campaign.md section 5).
GLOBAL_BATCH_SEQUENCES = 448
SEQUENCE_LENGTH = 4_096
GLOBAL_BATCH_TOKENS = GLOBAL_BATCH_SEQUENCES * SEQUENCE_LENGTH

# Total APUs released to the campaign once Praxis and Logos drop to 64 each for
# continued pretraining and post-training.
CAMPAIGN_APUS = 384
# One node held back for evaluation, canaries, and restarts.
CAMPAIGN_SPARE_APUS = 4

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


def _consecutive_primes(start: int, count: int) -> tuple[int, ...]:
    """``count`` distinct primes at or above ``start``.

    Distinct slot counts per hash head are what make the heads decorrelate:
    two heads with the same modulus collide on exactly the same key pairs, so
    the second head buys nothing.
    """

    found: list[int] = []
    candidate = max(3, start | 1)
    while len(found) < count:
        if _is_prime(candidate):
            found.append(candidate)
        candidate += 2
    return tuple(found)


# Proxy conditional memory: same shape as production, ~0.30B rows instead of
# 0.60B. Frozen rather than generated so the primary ladder's manifest is
# byte-stable across releases of this file.
_NGRAM_SLOTS_ORDER_2 = (
    292_969, 292_973, 292_979, 292_993, 293_021, 293_071, 293_081, 293_087,
)
_NGRAM_SLOTS_ORDER_3 = (
    293_093, 293_099, 293_107, 293_123, 293_129, 293_147, 293_149, 293_173,
)
_NGRAM_HASH_SEEDS = (
    2654435761, 2246822519, 3266489917, 668265263,
    374761393, 3550635117, 4251991749, 3043081225,
)


def _ngram_config(slots_per_head: int | None = None) -> NGramMemoryConfig:
    """Conditional memory sized for the model it sits in.

    The scaling ladder must not hold the table fixed while the neural core
    shrinks: at the smallest geometry a 0.30B table would be the majority of the
    stored model, and the scaling curve would mostly be measuring a constant
    lookup table.  Slot counts are therefore scaled so the table stays a roughly
    constant fraction of routed-expert capacity across the ladder.
    """

    if slots_per_head is None:
        order_2, order_3 = _NGRAM_SLOTS_ORDER_2, _NGRAM_SLOTS_ORDER_3
    else:
        primes = _consecutive_primes(int(slots_per_head), 16)
        order_2, order_3 = primes[:8], primes[8:]
    return NGramMemoryConfig(
        orders=(2, 3),
        tables_per_order=8,
        value_dim=64,
        slots_by_order={2: order_2, 3: order_3},
        hash_seeds=_NGRAM_HASH_SEEDS,
        table_mode="replicated",
        injection_layers=(2, 5),
    )


def _proxy_ngram_injection_layers(n_layers: int) -> tuple[int, ...]:
    """Place both conditional-memory injections inside the physical block."""

    if n_layers < 2:
        return (0,)
    if n_layers <= 2:
        return (0, 1)
    if n_layers <= 5:
        return (1, n_layers - 1)
    return (2, 5)


def _autotune() -> AutotuneConfig:
    return AutotuneConfig(
        micro_batch_sizes=(4, 2, 1),
        grad_accum_steps=(1, 2, 4, 7, 8),
        global_token_batch=GlobalTokenBatchBounds(
            GLOBAL_BATCH_TOKENS, GLOBAL_BATCH_TOKENS, GLOBAL_BATCH_TOKENS
        ),
        learning_rates=(0.00012, 0.00018, 0.00026),
        preferred_learning_rate=0.00018,
        gates=AutotuneGates(max_hbm_fraction=0.90),
    )


def proxy_config(
    *,
    world_size: int,
    ffn_mode: str = "moe",
    mhc_backend: str = "fused_required",
    mamba_backend: str = "fused_required",
    attention_backend: str = "varlen_fused_required",
    ngram_slots_per_head: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Metis16Config:
    """Parameter-matched shallow recurrent proxy for the MoRE ladder.

    Expert parallelism is fixed at 1: every rank replicates all routed experts
    and the whole job is data-parallel.  The model state is roughly 21GB against
    128GB of coherent HBM, so wide expert parallelism would buy nothing and cost
    an all-to-all -- and, worse for a science campaign, it would make row-to-row
    wall-clock differences partly an artifact of routing skew.
    """

    precision = PrecisionConfig()
    precision = replace(
        precision,
        fp8_scaling="blockwise",
        fp8_roles=tuple(
            role for role in precision.fp8_roles if role != "mhc_controller"
        ),
        bf16_roles=precision.bf16_roles + ("mhc_controller",),
        expert_collective_wire="bfloat16",
        require_fp8_validation=True,
        allow_bf16_fallback=True,
    )
    base: dict[str, Any] = {
        "schema": "metis.model-family/v1",
        "family": "ablation",
        "name": "MoRE-Proxy-S",
        "model_type": "metis16_more",
        "vocab_size": 65_536,
        "sequence_length": SEQUENCE_LENGTH,
        # The ablation campaign never extends context; the long-context claim
        # belongs to Praxis and Logos, not to a routing study.
        "final_context_length": SEQUENCE_LENGTH,
        "context_extension_train_length": SEQUENCE_LENGTH,
        # A shallow physical block is the architecture the experiment repeats.
        # At identical stored parameters and executed FLOPs, two wide layers
        # turn launch-bound micro-GEMMs into MI300A-sized contractions while
        # retaining one Mamba and one attention layer per recurrent pass.
        "d_model": 4_096,
        "n_layers": 2,
        "attention_indices": (1,),
        "n_heads": 64,
        "n_kv_heads": 16,
        "head_dim": 64,
        "attention_pass_lora_rank": 8,
        "attention_backend": attention_backend,
        "mamba_expand": 2,
        "mamba_d_state": 128,
        "mamba_d_conv": 4,
        "mamba_head_dim": 64,
        "mamba_ngroups": 16,
        "mamba_chunk_size": 256,
        "mamba_backend": mamba_backend,
        "n_streams": 4,
        "mhc_pass_embedding_dim": 64,
        "mhc_sinkhorn_iterations": 8,
        "mhc_backend": mhc_backend,
        "latent_dim": 2_048,
        "n_routed_experts": 72,
        "n_shared_experts": 1,
        "expert_intermediate_dim": 1_152,
        "min_routed_k": 1,
        "max_routed_k": 8,
        "target_mean_routed_k": 4.0,
        "world_size": int(world_size),
        "expert_parallel_size": 1,
        "expert_replicas": int(world_size),
        "expert_execution": "grouped_gemm",
        "max_passes": 5,
        "target_mean_passes": 2.0,
        # The trained policy uses easy / medium / hard depth levels. All three
        # are populated at an exact mean of two, avoiding both constant-depth
        # collapse and the tiny depth-4/5 tails that dominate launch overhead.
        "budgeted_depth_values": (1, 2, 3),
        "route_feature_dim": 256,
        "memory_dim": 256,
        "memory_heads": 4,
        "memory_gate_init": -6.0,
        "continuation_gate_init": 0.0,
        "router_z_loss_coefficient": 0.001,
        "expert_balance_coefficient": 0.0,
        "expert_balance_bias_update_rate": 0.001,
        "k_budget_coefficient": 1.0,
        "depth_budget_coefficient": 1.0,
        # The fixed coefficients above cannot hold either policy at its target:
        # measured on the canary, depth climbs from its intended 1.86 to the
        # 5.0 ceiling within nine steps and width sits at 7.3 against a target
        # of 4.0 from the first step. That is four times the budgeted compute,
        # and a MoRE-Core run whose adaptive-depth axis measures nothing. The
        # dual step turns them into an augmented Lagrangian that finds its own
        # strength.
        "budget_controller_rate": 1.0,
        "depth_budget_controller_rate": 20.0,
        # At the clamp boundary the gate's slope is d(sigmoid)/dz = 0.0177, so a
        # multiplier acting through it is attenuated about fifty-six fold. The
        # default ceiling of 1e3 is therefore an effective gain of about 18 --
        # less than the width budget reaches -- and the depth policy sat at the
        # ceiling against it. The penalty is proportional to the error, so a
        # high ceiling costs nothing once the constraint is met: it is a limit
        # on how hard the controller may pull, not on how hard it will.
        "budget_multiplier_limit": 1.0e5,
        "budget_controller_leak": 0.02,
        "continuation_entropy_coefficient": 0.001,
        "tie_embeddings": True,
        "moe_dispatch_chunks": 1,
        "activation_recompute_policy": "layer",
        "ffn_mode": ffn_mode,
        "ngram_memory": _ngram_config(ngram_slots_per_head),
        "precision": replace(
            precision,
            backend="auto",
            fp8_format="hybrid_e4m3_e5m2",
            # Transformer Engine 2.17 supplies a native gfx942 blockwise path,
            # which gives each row/column block its own scale instead of using
            # one tensor-wide amax. Production families retain delayed scaling
            # through PrecisionConfig's default; the ablation parity gate
            # validates this more aggressive execution policy per run.
        ),
        "autotune": _autotune(),
    }
    if overrides:
        base.update(dict(overrides))
    base["ngram_memory"] = replace(
        base["ngram_memory"],
        injection_layers=_proxy_ngram_injection_layers(int(base["n_layers"])),
    )
    if ffn_mode == "dense":
        # Applied *after* the overrides on purpose: a scaling-ladder geometry
        # carries an expert count, and a dense control must never inherit one no
        # matter what order the caller merged its dictionaries in.
        base.update(
            {
                "n_routed_experts": 0,
                "n_shared_experts": 0,
                "expert_replicas": int(world_size),
                "expert_parallel_size": 1,
            }
        )
    config = Metis16Config(**base)
    config.validate()
    return config


@dataclass(frozen=True)
class AblationSpec:
    """One row of the ladder."""

    index: int
    name: str
    title: str
    isolates: str
    apus: int
    micro_batch: int
    grad_accum: int
    ffn_mode: str = "moe"
    dense_ffn_intermediate_dim: int = 0
    d_model: int | None = None
    latent_dim: int | None = None
    continuation_mode: str = "adaptive"
    routed_k_mode: str = "adaptive"
    fixed_routed_k: int = 4
    pathway_mode: str = "per_pass"
    curriculum_max_passes: int | None = None
    depth_memory: bool = True
    iso_flop: bool = True
    muon_state_bits: int = 32
    notes: str = ""
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.muon_state_bits not in {8, 32}:
            raise ValueError(f"{self.name}: muon_state_bits must be 8 or 32.")
        product = self.apus * self.micro_batch * self.grad_accum
        if product != GLOBAL_BATCH_SEQUENCES:
            raise ValueError(
                f"{self.name}: apus*micro_batch*grad_accum = {product}, but every "
                f"row must consume exactly {GLOBAL_BATCH_SEQUENCES} sequences per "
                "optimizer step so all rows see identical token sets."
            )

    def model_config(self, **kwargs: Any) -> Metis16Config:
        overrides = dict(self.config_overrides)
        if self.ffn_mode == "dense":
            overrides["dense_ffn_intermediate_dim"] = self.dense_ffn_intermediate_dim
        if self.d_model is not None:
            overrides["d_model"] = self.d_model
            overrides["n_heads"] = self.d_model // 64
        if self.latent_dim is not None:
            overrides["latent_dim"] = self.latent_dim
        overrides["name"] = f"MoRE-Ablation-{self.index:02d}-{self.name}"
        fields, slots = _split_geometry(overrides)
        return proxy_config(
            world_size=self.apus,
            ffn_mode=self.ffn_mode,
            ngram_slots_per_head=slots,
            overrides=fields,
            **kwargs,
        )

    def curriculum(self, *, random_policy_seed: int = 0) -> CurriculumState:
        """Training-time routing policy for this row.

        ``memory_gate_scale=0`` is how MoRE-Core is separated from MoRE-RM: the
        recurrent depth-memory module keeps its parameters, so the two rows
        stay parameter-matched, but nothing it retrieves reaches the streams
        or routers. N-gram memory is a separate shared component and remains
        enabled in both rows.
        """

        return CurriculumState(
            continuation_mode=self.continuation_mode,
            routed_k_mode=self.routed_k_mode,
            fixed_routed_k=self.fixed_routed_k,
            pathway_mode=self.pathway_mode,
            max_passes=self.curriculum_max_passes,
            memory_gate_scale=1.0 if self.depth_memory else 0.0,
            ngram_gate_scale=1.0,
            stochastic_routing=True,
            temperature=1.0,
            target_mean_depth=2.0,
            target_mean_routed_k=4.0,
            random_policy_seed=random_policy_seed,
        )


def _split_geometry(
    geometry: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], int | None]:
    """Separate manifest fields from the private table-size hint."""

    fields = dict(geometry or {})
    slots = fields.pop("_ngram_slots_per_head", None)
    return fields, (int(slots) if slots is not None else None)


def _dense_candidate(
    intermediate: int,
    reference: Metis16Config,
    geometry: Mapping[str, Any] | None,
) -> Metis16Config:
    overrides, slots = _split_geometry(geometry)
    overrides["dense_ffn_intermediate_dim"] = intermediate
    return proxy_config(
        world_size=reference.world_size,
        ffn_mode="dense",
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
        ngram_slots_per_head=slots,
        overrides=overrides,
    )


def _solve_dense_intermediate(
    objective: str,
    *,
    passes: int,
    reference: Metis16Config,
    geometry: Mapping[str, Any] | None = None,
) -> int:
    """Size a dense control against the MoRE reference, rather than by hand.

    ``objective='stored'`` matches total stored parameters; ``objective='flops'``
    matches executed FLOPs per token at the given depth.  Both use the repo's
    own audited accounting -- ``logical_parameter_audit`` and
    ``estimate_hardware_flops`` -- so the paper's matching claim is derived from
    the same code that reports the training telemetry, not from a parallel
    hand-derivation that can silently drift.

    The search is over multiples of 64 to keep GEMM shapes well aligned; the
    residual mismatch is reported by :func:`dense_control_report` and belongs in
    the paper rather than being rounded away.
    """

    from metis_training.metrics import estimate_hardware_flops

    if objective == "stored":
        target = float(reference.logical_parameter_audit().stored_total)

        def measure(intermediate: int) -> float:
            return float(
                _dense_candidate(intermediate, reference, geometry)
                .logical_parameter_audit()
                .stored_total
            )

    elif objective == "flops":
        target = estimate_hardware_flops(
            reference,
            tokens=1,
            observed_mean_passes=float(reference.target_mean_passes),
            observed_mean_routed_k=float(reference.target_mean_routed_k),
        )

        def measure(intermediate: int) -> float:
            return estimate_hardware_flops(
                _dense_candidate(intermediate, reference, geometry),
                tokens=1,
                observed_mean_passes=float(passes),
            )

    else:
        raise ValueError("objective must be 'stored' or 'flops'")

    low, high = 64, 1 << 17
    while low < high:
        middle = ((low + high) // 2 + 63) // 64 * 64
        if middle >= high:
            break
        if measure(middle) < target:
            low = middle
        else:
            high = middle - 64 if measure(middle) > target else middle
        if high - low <= 64:
            break
    return max(64, (low // 64) * 64)


def _reference_config() -> Metis16Config:
    return proxy_config(
        world_size=28,
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
    )


# The dense controls cannot hold MoRE's stored parameters at MoRE's active
# parameters -- that asymmetry is exactly what the sparse-expert axis buys, and
# both numbers are reported rather than the flattering one.  Row 2 matches
# MoRE's stored parameters at depth 1; row 1 matches its executed FLOPs at
# depth 1; row 7 matches its executed FLOPs at depth 2.
_REFERENCE = _reference_config()
_DENSE_PARAM_MATCHED_INTERMEDIATE = _solve_dense_intermediate(
    "stored", passes=1, reference=_REFERENCE
)
_DENSE_FLOP_MATCHED_INTERMEDIATE = _solve_dense_intermediate(
    "flops", passes=1, reference=_REFERENCE
)
_DENSE_RECURSIVE_INTERMEDIATE = _solve_dense_intermediate(
    "flops", passes=2, reference=_REFERENCE
)


def dense_control_report() -> dict[str, dict[str, float]]:
    """Exact matching residuals for the dense controls, for the methods section."""

    from metis_training.metrics import estimate_hardware_flops

    def described(intermediate: int, passes: int) -> dict[str, float]:
        candidate = proxy_config(
            world_size=28,
            ffn_mode="dense",
            mhc_backend="torch_reference",
            mamba_backend="torch_reference",
            attention_backend="torch_reference",
            overrides={"dense_ffn_intermediate_dim": intermediate},
        )
        audit = candidate.logical_parameter_audit()
        return {
            "dense_ffn_intermediate_dim": float(intermediate),
            "stored_total": float(audit.stored_total),
            "active_per_pass": float(audit.active_per_pass_mean),
            "flops_per_token": estimate_hardware_flops(
                candidate, tokens=1, observed_mean_passes=float(passes)
            ),
        }

    reference_audit = _REFERENCE.logical_parameter_audit()
    return {
        "more_core": {
            "stored_total": float(reference_audit.stored_total),
            "active_per_pass": float(reference_audit.active_per_pass_mean),
            "flops_per_token": estimate_hardware_flops(
                _REFERENCE,
                tokens=1,
                observed_mean_passes=float(_REFERENCE.target_mean_passes),
                observed_mean_routed_k=float(_REFERENCE.target_mean_routed_k),
            ),
        },
        "dense_param_matched": described(_DENSE_PARAM_MATCHED_INTERMEDIATE, 1),
        "dense_flop_matched": described(_DENSE_FLOP_MATCHED_INTERMEDIATE, 1),
        "dense_recursive": described(_DENSE_RECURSIVE_INTERMEDIATE, 2),
    }


ABLATION_LADDER: tuple[AblationSpec, ...] = (
    AblationSpec(
        index=1,
        name="dense-flop-matched",
        title="Dense, FLOP-matched",
        isolates="dense reference at MoRE's executed compute",
        apus=28, micro_batch=4, grad_accum=4,
        ffn_mode="dense",
        dense_ffn_intermediate_dim=_DENSE_FLOP_MATCHED_INTERMEDIATE,
        continuation_mode="depth_one",
        notes="No recursion, no experts. The frontier point a reviewer expects.",
    ),
    AblationSpec(
        index=2,
        name="dense-param-matched",
        title="Dense, parameter-matched",
        isolates="dense reference at MoRE's stored parameters",
        apus=56, micro_batch=2, grad_accum=4,
        ffn_mode="dense",
        dense_ffn_intermediate_dim=_DENSE_PARAM_MATCHED_INTERMEDIATE,
        continuation_mode="depth_one",
        iso_flop=False,
        notes="Deliberately expensive per token; report against FLOPs, not steps.",
    ),
    AblationSpec(
        index=3,
        name="moe-k4",
        title="MoE k=4",
        isolates="sparse routing without recursion",
        apus=16, micro_batch=4, grad_accum=7,
        continuation_mode="depth_one",
        routed_k_mode="fixed", fixed_routed_k=4,
        depth_memory=False,
        iso_flop=False,
    ),
    AblationSpec(
        index=4,
        name="moe-k8",
        title="MoE k=8",
        isolates="wider single-pass MoE reference",
        apus=16, micro_batch=4, grad_accum=7,
        continuation_mode="depth_one",
        routed_k_mode="fixed", fixed_routed_k=8,
        depth_memory=False,
        iso_flop=False,
        notes=(
            "Not a compute match: k=4->8 adds 0.29 GFLOP/token while a second "
            "pass adds 1.76. Expert GEMMs are a minority of the block."
        ),
    ),
    AblationSpec(
        index=5,
        name="loop-fixed",
        title="Fixed LoopMoE",
        isolates="recursion at fixed depth and fixed k",
        apus=28, micro_batch=4, grad_accum=4,
        continuation_mode="fixed_max", curriculum_max_passes=2,
        routed_k_mode="fixed", fixed_routed_k=4,
        depth_memory=False,
        notes="Reimplements the published fixed-loop MoE design point in our backbone.",
    ),
    AblationSpec(
        index=6,
        name="loop-pathway-frozen",
        title="Loop, pathway frozen",
        isolates="PATHWAY: identical to row 5 except experts are chosen once",
        apus=28, micro_batch=4, grad_accum=4,
        continuation_mode="fixed_max", curriculum_max_passes=2,
        routed_k_mode="fixed", fixed_routed_k=4,
        pathway_mode="frozen",
        depth_memory=False,
        notes="Exactly iso-FLOP with row 5. The only evidence for axis three.",
    ),
    AblationSpec(
        index=7,
        name="mor-dense-ffn",
        title="MoR + dense FFN",
        isolates="adaptive depth without sparse experts",
        apus=28, micro_batch=4, grad_accum=4,
        ffn_mode="dense",
        dense_ffn_intermediate_dim=_DENSE_RECURSIVE_INTERMEDIATE,
        continuation_mode="budgeted",
        depth_memory=False,
    ),
    AblationSpec(
        index=8,
        name="mor-fixed-k",
        title="MoR + fixed-k MoE",
        isolates="DEPTH: adaptive depth against row 5's fixed depth",
        apus=28, micro_batch=4, grad_accum=4,
        continuation_mode="budgeted",
        routed_k_mode="fixed", fixed_routed_k=4,
        depth_memory=False,
    ),
    AblationSpec(
        index=9,
        name="fixed-depth-adaptive-k",
        title="Fixed depth, adaptive k",
        isolates="WIDTH: adaptive k against row 5's fixed k",
        apus=28, micro_batch=4, grad_accum=4,
        continuation_mode="fixed_max", curriculum_max_passes=2,
        routed_k_mode="budgeted",
        depth_memory=False,
    ),
    AblationSpec(
        index=10,
        name="more-core",
        title="MoRE-Core",
        isolates="all three axes together",
        # The proxy fits eight 4K sequences per MI300A at the target depth.
        # Holding the global batch fixed while halving accumulation amortizes
        # Python, recompute setup, and loader overhead over twice the tokens.
        apus=28, micro_batch=8, grad_accum=2,
        continuation_mode="budgeted",
        routed_k_mode="budgeted",
        depth_memory=False,
    ),
    AblationSpec(
        index=11,
        name="more-rm",
        title="MoRE-RM",
        isolates="route-typed recurrent depth memory",
        apus=32, micro_batch=2, grad_accum=7,
        continuation_mode="budgeted",
        routed_k_mode="budgeted",
        depth_memory=True,
    ),
    AblationSpec(
        index=12,
        name="random-k",
        title="Random-k control",
        isolates="is the LEARNED width policy doing anything?",
        apus=28, micro_batch=4, grad_accum=4,
        continuation_mode="budgeted",
        routed_k_mode="random",
        depth_memory=False,
        notes="Maximum-entropy width distribution at the same mean budget.",
    ),
    AblationSpec(
        index=13,
        name="random-depth",
        title="Random-depth control",
        isolates="is the LEARNED depth policy doing anything?",
        apus=28, micro_batch=4, grad_accum=4,
        continuation_mode="random",
        routed_k_mode="budgeted",
        depth_memory=False,
        notes="Memoryless halt tuned to the same mean depth.",
    ),
)


# =========================================================================
# Wave 2 -- scaling ladder
#
# One size is a data point, three sizes are a trend.  Running the four
# archetypes at two smaller geometries turns "MoRE beats its baselines at 1.8B"
# into a slope, with Praxis and Logos as the fourth and fifth points on the same
# curve.  The geometries scale d_model, latent width, and expert count together
# so the aspect ratio of the model is held roughly constant; scaling only one of
# them would confound size with shape.

_SCALE_GEOMETRIES: dict[str, dict[str, Any]] = {
    # ~1/4 Praxis.  n_heads follows d_model at head_dim 64; n_kv_heads keeps the
    # 4:1 GQA ratio; mamba_ngroups divides the Mamba head count.
    "xs": {
        "d_model": 1_280,
        "n_heads": 20,
        "n_kv_heads": 5,
        "n_layers": 8,
        "attention_indices": (2, 5),
        "latent_dim": 640,
        "expert_intermediate_dim": 320,
        "n_routed_experts": 64,
        "mamba_ngroups": 8,
        "_ngram_slots_per_head": 79_883,
    },
    # ~1/8 Praxis.
    "xxs": {
        "d_model": 896,
        "n_heads": 14,
        "n_kv_heads": 7,
        "n_layers": 6,
        "attention_indices": (1, 4),
        "latent_dim": 448,
        "expert_intermediate_dim": 224,
        "n_routed_experts": 48,
        "mamba_ngroups": 7,
        "_ngram_slots_per_head": 22_013,
    },
}

# Archetypes carried down the scaling ladder: the dense reference, the
# single-pass sparse reference, and the two headline MoRE rows.
_SCALING_ARCHETYPES = ("dense-param-matched", "moe-k4", "more-core", "more-rm")


def scaled_ablation_ladder(scale: str) -> tuple[AblationSpec, ...]:
    """The whole wave-1 ladder at one of the scaling-ladder geometries.

    Wave 2 already carries four archetypes down to XS and XXS; this carries all
    thirteen rows, which is what an allocation window too small for the primary
    geometry needs. Every row moves together -- ``d_model``, latent width, layer
    count, expert count and the N-gram table all come from the same geometry --
    so the comparison between rows is untouched. Only the point on the size axis
    moves, and the campaign already treats that axis as something to report.

    The dense controls are re-solved rather than rescaled. Their whole purpose
    is to match MoRE's stored parameters or its executed FLOPs at the geometry
    they actually run at, and a hand-scaled intermediate width would quietly
    stop matching either.
    """

    if scale not in _SCALE_GEOMETRIES:
        raise ValueError(f"Unknown scale {scale!r}; expected one of {sorted(_SCALE_GEOMETRIES)}.")
    fields, _slots = _split_geometry(_SCALE_GEOMETRIES[scale])
    geometry = dict(_SCALE_GEOMETRIES[scale])
    dense_by_objective = {
        ("stored", 1): _scaled_dense_intermediate(scale, "stored", 1),
        ("flops", 1): _scaled_dense_intermediate(scale, "flops", 1),
        ("flops", 2): _scaled_dense_intermediate(scale, "flops", 2),
    }
    scaled: list[AblationSpec] = []
    for spec in ABLATION_LADDER:
        overrides = dict(spec.config_overrides or {})
        overrides.update(geometry)
        dense_intermediate = spec.dense_ffn_intermediate_dim
        if spec.ffn_mode == "dense":
            # Which quantity this control matches is a property of the row, and
            # it has to be re-solved at the new geometry to go on matching it.
            if spec.name == "dense-param-matched":
                key = ("stored", 1)
            elif spec.continuation_mode == "depth_one":
                key = ("flops", 1)
            else:
                key = ("flops", 2)
            dense_intermediate = dense_by_objective[key]
        scaled.append(
            replace(
                spec,
                dense_ffn_intermediate_dim=dense_intermediate,
                config_overrides=overrides,
                notes=f"{spec.notes} Run at the {scale.upper()} geometry.",
            )
        )
    return tuple(scaled)


def _scaled_dense_intermediate(scale: str, objective: str, passes: int) -> int:
    fields, slots = _split_geometry(_SCALE_GEOMETRIES[scale])
    reference = proxy_config(
        world_size=16,
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
        ngram_slots_per_head=slots,
        overrides=fields,
    )
    return _solve_dense_intermediate(
        objective,
        passes=passes,
        reference=reference,
        geometry=_SCALE_GEOMETRIES[scale],
    )


# Allocation per (scale, archetype), chosen so every wave-2 row finishes within
# about half an hour of the others.  Each count divides 448, so the global batch
# is identical to wave 1 and the scaling curve is directly comparable to the main
# ladder rather than merely similar.  ``(apus, micro_batch, grad_accum)``.
_SCALING_ALLOCATION: dict[tuple[str, str], tuple[int, int, int]] = {
    ("xs", "dense-param-matched"): (64, 1, 7),
    ("xs", "moe-k4"): (32, 2, 7),
    ("xs", "more-core"): (56, 8, 1),
    ("xs", "more-rm"): (56, 2, 4),
    ("xxs", "dense-param-matched"): (28, 4, 4),
    ("xxs", "moe-k4"): (16, 4, 7),
    ("xxs", "more-core"): (28, 8, 2),
    ("xxs", "more-rm"): (28, 4, 4),
}


def _scaling_specs() -> tuple[AblationSpec, ...]:
    specs: list[AblationSpec] = []
    index = 20
    for scale in ("xs", "xxs"):
        geometry = dict(_SCALE_GEOMETRIES[scale])
        for archetype in _SCALING_ARCHETYPES:
            apus, micro_batch, grad_accum = _SCALING_ALLOCATION[(scale, archetype)]
            base = spec_by_name(archetype, ladder=ABLATION_LADDER)
            overrides = dict(geometry)
            dense_intermediate = base.dense_ffn_intermediate_dim
            if base.ffn_mode == "dense":
                dense_intermediate = _scaled_dense_intermediate(
                    scale, "stored", 1
                )
            specs.append(
                AblationSpec(
                    index=index,
                    name=f"{archetype}-{scale}",
                    title=f"{base.title} ({scale.upper()})",
                    isolates=f"scaling point: {base.isolates}",
                    apus=apus,
                    micro_batch=micro_batch,
                    grad_accum=grad_accum,
                    ffn_mode=base.ffn_mode,
                    dense_ffn_intermediate_dim=dense_intermediate,
                    continuation_mode=base.continuation_mode,
                    routed_k_mode=base.routed_k_mode,
                    fixed_routed_k=base.fixed_routed_k,
                    pathway_mode=base.pathway_mode,
                    curriculum_max_passes=base.curriculum_max_passes,
                    depth_memory=base.depth_memory,
                    iso_flop=False,
                    notes=f"Scaling ladder point at {scale}; pairs with {archetype}.",
                    config_overrides=overrides,
                )
            )
            index += 1
    return tuple(specs)


# =========================================================================
# Wave 3 -- paired seeds
#
# A second seed is insurance against one lucky headline result, not a different
# model design.  Only the three rows the abstract quotes need it.

_SEEDED_ROWS = (
    "dense-param-matched",
    "more-core",
    "more-rm",
    # The pathway axis rests on a single comparison, rows 5 and 6, and it is the
    # one place the campaign has no corroborating row. Those two are also where
    # an effect is most likely to be small: they are exactly iso-FLOP and differ
    # only in whether experts are rechosen per pass. A one-seed difference there
    # cannot be distinguished from initialisation noise, so both get a paired
    # repeat. The architecture comparisons above have large effects; this one
    # may not.
    "loop-fixed",
    "loop-pathway-frozen",
)
SECOND_SEED = 27_182_818

# Sequences per global batch are fixed campaign-wide, so apus x micro x accum
# must come to 448 for every row. dense-param-matched keeps the wide allocation
# because it is by far the most expensive row -- 12.96 GF/tok against 7.30 --
# and it is the one that would run out of wall clock first at low MFU. The rest
# take half of it, which keeps the wave inside the campaign's APU budget.
# micro_batch follows each row's wave-1 choice: more-rm carries depth memory and
# was given the smaller micro batch there.
_SEED_ALLOCATION: dict[str, tuple[int, int, int]] = {
    "dense-param-matched": (112, 1, 4),
    "more-core": (56, 8, 1),
    "more-rm": (56, 2, 4),
    "loop-fixed": (56, 4, 2),
    "loop-pathway-frozen": (56, 4, 2),
}


def _seed_specs() -> tuple[AblationSpec, ...]:
    specs: list[AblationSpec] = []
    for offset, name in enumerate(_SEEDED_ROWS):
        base = spec_by_name(name, ladder=ABLATION_LADDER)
        apus, micro, accum = _SEED_ALLOCATION[name]
        specs.append(
            AblationSpec(
                index=40 + offset,
                name=f"{name}-seed2",
                title=f"{base.title} (seed 2)",
                isolates=f"paired repeat of {name}",
                apus=apus,
                micro_batch=micro,
                grad_accum=accum,
                ffn_mode=base.ffn_mode,
                dense_ffn_intermediate_dim=base.dense_ffn_intermediate_dim,
                continuation_mode=base.continuation_mode,
                routed_k_mode=base.routed_k_mode,
                fixed_routed_k=base.fixed_routed_k,
                pathway_mode=base.pathway_mode,
                curriculum_max_passes=base.curriculum_max_passes,
                depth_memory=base.depth_memory,
                iso_flop=base.iso_flop,
                notes=(
                    f"Second seed for {name}; identical data order, different "
                    "initialization. Launch with --seed "
                    f"{SECOND_SEED}."
                ),
                config_overrides=dict(base.config_overrides),
            )
        )
    return tuple(specs)


def spec_by_name(
    name: str,
    *,
    ladder: tuple[AblationSpec, ...] | None = None,
) -> AblationSpec:
    for spec in ladder if ladder is not None else ALL_SPECS:
        if spec.name == name:
            return spec
    known = ", ".join(
        spec.name for spec in (ladder if ladder is not None else ALL_SPECS)
    )
    raise KeyError(f"Unknown ablation row {name!r}. Known rows: {known}")


SCALING_LADDER: tuple[AblationSpec, ...] = _scaling_specs()
SEED_LADDER: tuple[AblationSpec, ...] = _seed_specs()

WAVES: dict[str, tuple[AblationSpec, ...]] = {
    "1": ABLATION_LADDER,
    "2": SCALING_LADDER,
    "3": SEED_LADDER,
}
ALL_SPECS: tuple[AblationSpec, ...] = (
    ABLATION_LADDER + SCALING_LADDER + SEED_LADDER
)



def wave_for_row(name: str) -> str:
    for wave, specs in WAVES.items():
        if any(spec.name == name for spec in specs):
            return wave
    raise KeyError(f"Unknown ablation row {name!r}")


def validate_allocation(
    specs: tuple[AblationSpec, ...] | None = None,
    *,
    total_apus: int = CAMPAIGN_APUS,
    spare_apus: int = CAMPAIGN_SPARE_APUS,
) -> dict[str, int]:
    """Fail loudly if a wave does not fit, rather than at submission time."""

    specs = ABLATION_LADDER if specs is None else specs
    allocated = sum(spec.apus for spec in specs)
    budget = total_apus - spare_apus
    if allocated > budget:
        raise ValueError(
            f"Ablation wave requests {allocated} APUs but only {budget} are "
            f"available ({total_apus} minus {spare_apus} held for evaluation)."
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("Ablation row names must be unique.")
    for spec in specs:
        if spec.apus % 4:
            raise ValueError(
                f"{spec.name} requests {spec.apus} APUs, which is not a whole "
                "number of 4-APU nodes."
            )
    return {
        "rows": len(specs),
        "allocated_apus": allocated,
        "spare_apus": total_apus - allocated,
        "global_batch_sequences": GLOBAL_BATCH_SEQUENCES,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
    }


__all__ = [
    "ABLATION_LADDER",
    "AblationSpec",
    "CAMPAIGN_APUS",
    "CAMPAIGN_SPARE_APUS",
    "GLOBAL_BATCH_SEQUENCES",
    "GLOBAL_BATCH_TOKENS",
    "SEQUENCE_LENGTH",
    "proxy_config",
    "spec_by_name",
    "validate_allocation",
]
