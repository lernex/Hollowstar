from __future__ import annotations

import contextlib
import contextvars
import hashlib
import importlib
import io
import math
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .model_config import PrecisionConfig
from .precision_plan import (
    build_precision_role_inventory,
    build_precision_role_plan,
    exact_precision_role_specs,
    representative_probe_rows,
)


_DYNAMIC_ROW_LINEAR_TYPES: dict[type[nn.Module], type[nn.Module]] = {}
# Transformer Engine contracts over the input's last dimension and requires it
# to be a multiple of sixteen; the row padding above covers the leading dims.
_FP8_CONTRACTION_MULTIPLE = 16
# Packed row counts are rounded up to a geometric bucket rather than to the
# bare FP8 stride; see :func:`bucketed_row_count`. Three means no call is
# padded by more than an eighth of itself, and eight buckets cover each octave.
_ROW_BUCKET_SHIFT = 3


@dataclass
class _ActivationCheckpointFrame:
    """Per-checkpoint FP8 state, independent of backward traversal order."""

    slots: list[dict[tuple[int, str, str], torch.Tensor]]
    recompute_cursor: int = 0


def _fp8_state_tensors(module: nn.Module) -> dict[tuple[int, str, str], torch.Tensor]:
    """Address every mutable tensor in TE delayed-scaling metadata."""

    tensors: dict[tuple[int, str, str], torch.Tensor] = {}
    for child in module.modules():
        metadata = getattr(child, "fp8_meta", None)
        if not isinstance(metadata, Mapping):
            continue
        for name, value in metadata.items():
            if isinstance(value, torch.Tensor):
                tensors[(id(child), str(name), "")] = value
                continue
            for attribute in ("scale", "scale_inv", "amax_history"):
                tensor = getattr(value, attribute, None)
                if isinstance(tensor, torch.Tensor):
                    tensors[(id(child), str(name), attribute)] = tensor
    return tensors


def _initialize_fp8_state_tensors(module: nn.Module) -> None:
    """Materialize TE Linear metadata before taking a recompute snapshot."""

    for child in module.modules():
        if not isinstance(getattr(child, "fp8_meta", None), Mapping):
            continue
        initializer = getattr(child, "init_fp8_metadata", None)
        if callable(initializer) and not _fp8_state_tensors(child):
            # Every FP8 module built by this trainer is a TE Linear and owns one
            # GEMM. Initializing under autocast reproduces prepare_forward's
            # default metadata without executing the GEMM.
            initializer(num_gemms=1)


def _clone_fp8_state(
    module: nn.Module,
) -> dict[tuple[int, str, str], torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in _fp8_state_tensors(module).items()
    }


@torch.no_grad()
def _restore_fp8_state(
    module: nn.Module,
    snapshot: Mapping[tuple[int, str, str], torch.Tensor],
) -> None:
    current = _fp8_state_tensors(module)
    if set(current) != set(snapshot):
        raise RuntimeError(
            "Transformer Engine FP8 metadata inventory changed during "
            "activation recomputation"
        )
    for key, saved in snapshot.items():
        target = current[key]
        if target.shape != saved.shape or target.dtype != saved.dtype:
            raise RuntimeError(
                "Transformer Engine FP8 metadata shape/dtype changed during "
                "activation recomputation"
            )
        target.copy_(saved)


def bucketed_row_count(
    row_count: int,
    *,
    multiple: int = _FP8_CONTRACTION_MULTIPLE,
    shift: int = _ROW_BUCKET_SHIFT,
) -> int:
    """Round a packed row count up to one of a small number of buckets.

    Padding to the bare sixteen-row FP8 requirement leaves the row count free
    to take four thousand distinct values, and pass packing makes it take a
    different one on every step. hipBLASLt -- which Transformer Engine uses for
    its GEMMs regardless of what the aten surface is told to prefer -- searches
    for a kernel per shape and caches the answer per shape, so a shape it never
    sees twice is a search it never amortizes. Measured on one MI300A: 0.91 ms
    per forward and backward at a repeated row count, **908 ms** at row counts
    varying by sixteen, and 0.70 ms at row counts rounded to a thousand.

    Bucketing geometrically rather than to a fixed stride keeps the waste
    bounded at both ends: the step is the largest power of two no greater than
    ``row_count`` divided by ``2**shift``, so no call is padded by more than
    ``1/2**shift`` of itself, and the whole range of row counts collapses to
    about ``2**shift`` buckets per octave.

    The padding is arithmetically free. Rows of zeros contribute nothing to a
    bias-free projection, they cannot raise an absolute maximum, and the padded
    outputs are sliced off before anything reads them. It is not bitwise free:
    changing M lets the GEMM block and accumulate differently. That was already
    true of the sixteen-row padding this replaces, and it moves every row of
    the campaign the same way.
    """

    if row_count <= 0:
        return multiple
    step = max(multiple, 1 << max(0, row_count.bit_length() - 1 - shift))
    return -(-row_count // step) * step


def _dynamic_row_linear_type(base: type[nn.Module]) -> type[nn.Module]:
    """Return a TE Linear subclass that pads arbitrary packed row counts."""

    cached = _DYNAMIC_ROW_LINEAR_TYPES.get(base)
    if cached is not None:
        return cached

    class DynamicRowLinear(base):  # type: ignore[misc, valid-type]
        metis_dynamic_row_multiple = 16
        metis_dynamic_row_bucket_shift = _ROW_BUCKET_SHIFT

        def forward(self, values: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            # ``nn.Linear`` accepts a bare feature vector and returns one, and
            # this class stands in for it under FP8, so it has to accept the
            # same ranks. The mHC controller is the case that matters: its
            # pass embedding is per pass, not per token, so it arrives 1D. The
            # flatten below already handles that correctly -- reshape(-1, D)
            # makes it a single row -- and rejecting it only made the FP8 and
            # BF16 paths disagree about what is a legal input.
            if values.ndim < 1:
                raise RuntimeError(
                    "Transformer Engine Linear input must have a feature dimension"
                )
            leading = values.shape[:-1]
            flattened = values.reshape(-1, values.shape[-1])
            row_count = int(flattened.shape[0])
            padded_rows = bucketed_row_count(
                row_count,
                multiple=self.metis_dynamic_row_multiple,
                shift=self.metis_dynamic_row_bucket_shift,
            ) - row_count
            if padded_rows:
                flattened = F.pad(flattened, (0, 0, 0, padded_rows))
            result = super().forward(flattened, *args, **kwargs)
            if not isinstance(result, torch.Tensor):
                raise RuntimeError(
                    "Metis FP8 Linear requires Transformer Engine to return one tensor"
                )
            result = result[:row_count]
            return result.reshape(*leading, result.shape[-1])

    DynamicRowLinear.__name__ = f"MetisDynamicRow{base.__name__}"
    DynamicRowLinear.__qualname__ = DynamicRowLinear.__name__
    _DYNAMIC_ROW_LINEAR_TYPES[base] = DynamicRowLinear
    return DynamicRowLinear


def _fp8_metadata_digest(module: nn.Module) -> str:
    """Hash TE delayed-scaling metadata after an autocast context exits."""

    metadata = {
        name: value
        for name, value in module.state_dict().items()
        if "extra_state" in name or "amax" in name or "scale" in name
    }
    if not metadata:
        fp8_meta = getattr(module, "fp8_meta", None)
        if fp8_meta is not None:
            metadata = {"fp8_meta": fp8_meta}
    if not metadata:
        raise RuntimeError(
            "Transformer Engine exposed no inspectable FP8 amax/scale metadata"
        )
    buffer = io.BytesIO()
    try:
        torch.save(metadata, buffer)
    except BaseException as exc:
        raise RuntimeError(
            "Transformer Engine FP8 metadata could not be serialized"
        ) from exc
    return hashlib.sha256(buffer.getvalue()).hexdigest()


@dataclass(frozen=True)
class PrecisionAudit:
    requested_profile: str
    effective_profile: str
    torch_version: str
    hip_version: str | None
    device_name: str
    device_arch: str
    transformer_engine_available: bool
    transformer_engine_version: str | None
    fp8_format: str | None
    fp8_scaling: str | None
    fp8_linear_roles: tuple[str, ...]
    bf16_roles: tuple[str, ...]
    fp32_roles: tuple[str, ...]
    fallback_reason: str | None
    precision_role_plan_sha256: str | None
    measured_role_dtypes: dict[str, str]
    execution_role_dtypes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _device_identity(device: torch.device) -> tuple[str, str]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return str(device), ""
    props = torch.cuda.get_device_properties(device)
    name = str(getattr(props, "name", "unknown"))
    arch = str(
        getattr(props, "gcnArchName", "")
        or getattr(props, "gcn_arch_name", "")
        or os.environ.get("PYTORCH_ROCM_ARCH", "")
    )
    return name, arch


def _load_transformer_engine() -> tuple[Any | None, Any | None, str | None]:
    try:
        te = importlib.import_module("transformer_engine.pytorch")
        recipe = importlib.import_module("transformer_engine.common.recipe")
        package = importlib.import_module("transformer_engine")
        return te, recipe, str(getattr(package, "__version__", "unknown"))
    except (ImportError, OSError, RuntimeError):
        return None, None, None


class PrecisionPolicy:
    """Own the executable numerical policy for one training process.

    FP8 is only claimed when Transformer Engine is importable on a ROCm GPU.
    Eligible projections are constructed as TE Linear modules; wrapping ordinary
    ``torch.nn.Linear`` in an FP8 context would not constitute FP8 execution.
    Sensitive state, routing, reductions, optimizer state, and master weights
    remain explicitly outside this policy's FP8 surface.
    """

    def __init__(
        self,
        config: PrecisionConfig,
        *,
        requested_profile: str,
        device: torch.device,
        production: bool,
        permit_fallback: bool | None = None,
        measured_role_dtypes: Mapping[str, str] | None = None,
        precision_role_plan_sha256: str | None = None,
    ) -> None:
        config.validate()
        profile = requested_profile.lower()
        if profile not in {"fp8", "bf16"}:
            raise ValueError("precision profile must be fp8 or bf16")
        self.config = config
        self.requested_profile = profile
        self.device = device
        self.production = bool(production)
        self._te, self._recipe_module, self._te_version = _load_transformer_engine()
        self._fp8_recipe: Any | None = None
        self._fallback_reason: str | None = None
        self._force_bf16_depth = 0
        self._activation_recompute_phase: contextvars.ContextVar[
            tuple[_ActivationCheckpointFrame, bool] | None
        ] = contextvars.ContextVar(
                f"metis_fp8_activation_recompute_{id(self)}",
                default=None,
        )
        self._measured_role_dtypes = (
            {
                str(role): str(dtype)
                for role, dtype in measured_role_dtypes.items()
            }
            if measured_role_dtypes is not None
            else None
        )
        self.precision_role_plan_sha256 = precision_role_plan_sha256
        if self._measured_role_dtypes is not None:
            if (
                not self._measured_role_dtypes
                or any(
                    dtype not in {"fp8", "bf16"}
                    for dtype in self._measured_role_dtypes.values()
                )
            ):
                raise RuntimeError("Precision role plan contains an invalid dtype map")
            if not isinstance(precision_role_plan_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", precision_role_plan_sha256
            ):
                raise RuntimeError("Precision role plan hash is missing or invalid")
        elif precision_role_plan_sha256 is not None:
            raise RuntimeError("Precision role plan hash was supplied without its role map")
        elif production:
            raise RuntimeError(
                "Production precision requires a sealed exact-role dtype plan"
            )

        hip_version = getattr(torch.version, "hip", None)
        name, arch = _device_identity(device)
        rocm_gpu = device.type == "cuda" and bool(hip_version)
        te_ready = self._te is not None and self._recipe_module is not None
        fallback_allowed = (
            config.allow_bf16_fallback if permit_fallback is None else bool(permit_fallback)
        )
        if profile == "fp8" and not rocm_gpu:
            self._fallback_reason = "FP8 requires a ROCm-visible GPU"
        elif profile == "fp8" and not te_ready:
            self._fallback_reason = "ROCm Transformer Engine is unavailable"

        if profile == "fp8" and self._fallback_reason:
            if production or not fallback_allowed:
                raise RuntimeError(
                    f"Cannot honor requested FP8 profile: {self._fallback_reason}. "
                    "The launcher must select a measured BF16 candidate explicitly."
                )
            self.effective_profile = "bf16"
        else:
            self.effective_profile = profile

        if self.effective_profile == "fp8":
            if (
                self._measured_role_dtypes is not None
                and "fp8" not in set(self._measured_role_dtypes.values())
            ):
                raise RuntimeError(
                    "Requested FP8 profile has no measured throughput-positive FP8 role"
                )
            self._fp8_recipe = self._build_fp8_recipe()
        measured_map = dict(sorted((self._measured_role_dtypes or {}).items()))
        execution_map = (
            {role: "bf16" for role in measured_map}
            if self.effective_profile == "bf16"
            else measured_map
        )
        self.audit = PrecisionAudit(
            requested_profile=profile,
            effective_profile=self.effective_profile,
            torch_version=str(torch.__version__),
            hip_version=str(hip_version) if hip_version else None,
            device_name=name,
            device_arch=arch,
            transformer_engine_available=te_ready,
            transformer_engine_version=self._te_version,
            fp8_format=config.fp8_format if self.effective_profile == "fp8" else None,
            fp8_scaling=(
                config.fp8_scaling if self.effective_profile == "fp8" else None
            ),
            fp8_linear_roles=tuple(
                role for role, dtype in execution_map.items() if dtype == "fp8"
            )
            if measured_map
            else tuple(config.fp8_roles),
            bf16_roles=tuple(config.bf16_roles),
            fp32_roles=tuple(config.fp32_roles),
            fallback_reason=self._fallback_reason,
            precision_role_plan_sha256=precision_role_plan_sha256,
            measured_role_dtypes=measured_map,
            execution_role_dtypes=execution_map,
        )

    @property
    def fp8_enabled(self) -> bool:
        return self.effective_profile == "fp8"

    def is_fp8_role(self, role: str) -> bool:
        if self._measured_role_dtypes is not None:
            if role not in self._measured_role_dtypes:
                raise RuntimeError(
                    f"Linear role {role!r} is absent from the sealed precision plan"
                )
            return (
                self.fp8_enabled
                and self._measured_role_dtypes[role] == "fp8"
            )
        lowered = role.lower()
        return self.fp8_enabled and any(
            token in lowered or lowered in token for token in self.config.fp8_roles
        )

    def linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        role: str,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a projection on the declared precision surface."""

        if self.is_fp8_role(role):
            if self._te is None:
                raise RuntimeError("FP8 role requested without Transformer Engine")
            # TE contracts over the input's last dimension and requires it to be
            # a multiple of sixteen. A role declared FP8 at a width that cannot
            # be executed is a manifest error, and the cheap place to find it is
            # here -- at construction, on one rank -- rather than in the first
            # forward of an allocated multi-node job.
            if in_features % _FP8_CONTRACTION_MULTIPLE:
                raise RuntimeError(
                    f"Linear role {role!r} is declared FP8 but contracts over "
                    f"{in_features} features; Transformer Engine requires a "
                    f"multiple of {_FP8_CONTRACTION_MULTIPLE}. Declare the role "
                    "in bf16_roles instead."
                )
            linear_type = _dynamic_row_linear_type(self._te.Linear)
            return linear_type(in_features, out_features, bias=bias, **kwargs)
        return nn.Linear(in_features, out_features, bias=bias, **kwargs)

    def grouped_linear(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        role: str,
        **kwargs: Any,
    ) -> nn.Module | None:
        """Create one projection that contracts every expert in a single GEMM.

        Returns ``None`` when no grouped implementation is available, which
        tells the caller to build the portable stacked-weight bank instead.

        Unlike :meth:`linear` this does not consult the role's declared
        precision. The grouped bank runs on the BF16 surface regardless -- see
        :class:`~metis_training.model.GroupedSwiGLUExperts` -- so the only
        question here is whether Transformer Engine can supply the grouped
        kernel, and TE's grouped GEMM is available on the BF16 surface whether
        or not the role was declared FP8.
        """

        if self._te is None:
            return None
        grouped_type = getattr(self._te, "GroupedLinear", None)
        if grouped_type is None:
            return None
        return grouped_type(num_gemms, in_features, out_features, bias=bias, **kwargs)

    @property
    def grouped_row_multiple(self) -> int:
        """Rows each expert segment is padded to before the grouped GEMM."""

        return _FP8_CONTRACTION_MULTIPLE

    def _build_fp8_recipe(self) -> Any:
        assert self._recipe_module is not None
        recipe_mod = self._recipe_module
        format_enum = getattr(recipe_mod, "Format", None)
        format_name = self.config.fp8_format.lower()
        if format_enum is None:
            raise RuntimeError("Transformer Engine recipe.Format is unavailable")
        if format_name == "e4m3fnuz":
            # AMD's FNUZ hardware encoding is selected by the ROCm backend.
            # TE's public recipe still describes the mathematical E4M3 format.
            fp8_format = getattr(format_enum, "E4M3")
        elif format_name == "hybrid_e4m3_e5m2":
            # Use E4M3 on the forward surface and E5M2 for backward gradients.
            # ROCm Transformer Engine selects the CDNA3 FNUZ encodings behind
            # the public HYBRID recipe.
            fp8_format = getattr(format_enum, "HYBRID")
        else:
            fp8_format = getattr(format_enum, "E4M3")
        if self.config.fp8_scaling == "current":
            current = getattr(recipe_mod, "Float8CurrentScaling", None)
            if current is None:
                raise RuntimeError(
                    "Transformer Engine Float8CurrentScaling is unavailable"
                )
            return current(fp8_format=fp8_format)
        delayed = getattr(recipe_mod, "DelayedScaling", None)
        if delayed is None:
            raise RuntimeError("Transformer Engine DelayedScaling recipe is unavailable")
        return delayed(
            margin=0,
            fp8_format=fp8_format,
            amax_history_len=1024,
            amax_compute_algo="max",
        )

    @contextlib.contextmanager
    def execution_context(self, *, module: nn.Module | None = None) -> Iterator[None]:
        if self.device.type == "cuda":
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif self.device.type == "cpu":
            amp = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            amp = contextlib.nullcontext()
        with amp:
            if not self.fp8_enabled or self._force_bf16_depth:
                yield
                return
            assert self._te is not None and self._fp8_recipe is not None
            fp8_context = getattr(self._te, "fp8_autocast", None)
            if fp8_context is not None:
                fp8_kwargs = {
                    "enabled": True,
                    "fp8_recipe": self._fp8_recipe,
                }
            else:
                fp8_context = getattr(self._te, "autocast", None)
                fp8_kwargs = {
                    "enabled": True,
                    "recipe": self._fp8_recipe,
                }
            if fp8_context is None:
                raise RuntimeError("Transformer Engine exposes no FP8 autocast context")
            checkpoint_phase = self._activation_recompute_phase.get()
            restore_after: dict[tuple[int, str, str], torch.Tensor] | None = None
            try:
                with fp8_context(**fp8_kwargs):
                    if checkpoint_phase is not None:
                        if module is None:
                            raise RuntimeError(
                                "FP8 activation recompute requires the exact TE "
                                "module surface"
                            )
                        _initialize_fp8_state_tensors(module)
                        frame, recompute_phase = checkpoint_phase
                        if recompute_phase:
                            if frame.recompute_cursor >= len(frame.slots):
                                raise RuntimeError(
                                    "FP8 activation recompute executed an unexpected "
                                    "logical surface"
                                )
                            restore_after = _clone_fp8_state(module)
                            original_state = frame.slots[frame.recompute_cursor]
                            frame.recompute_cursor += 1
                            _restore_fp8_state(module, original_state)
                        else:
                            frame.slots.append(_clone_fp8_state(module))
                    yield
            finally:
                # A recomputed TE forward is allowed to perform its normal
                # delayed-scaling collective, but it must not become the next
                # optimizer step's numerical state. Restore the exact
                # post-original-forward tensors after FP8 autocast has reduced
                # and rotated its temporary amax buffer.
                if restore_after is not None:
                    assert module is not None
                    _restore_fp8_state(module, restore_after)

    @contextlib.contextmanager
    def _activation_checkpoint_phase(
        self,
        *,
        frame: _ActivationCheckpointFrame,
        recompute_phase: bool,
    ) -> Iterator[None]:
        if recompute_phase:
            frame.recompute_cursor = 0
        token = self._activation_recompute_phase.set(
            (frame, bool(recompute_phase))
        )
        try:
            yield
        finally:
            self._activation_recompute_phase.reset(token)
            if recompute_phase and frame.recompute_cursor != len(frame.slots):
                raise RuntimeError(
                    "FP8 activation recompute did not replay every logical surface"
                )

    def activation_checkpoint_context_fn(
        self,
    ) -> tuple[contextlib.AbstractContextManager[Any], contextlib.AbstractContextManager[Any]]:
        """Contexts for ``torch.utils.checkpoint`` with per-surface TE state.

        The outer checkpoint callable may contain many independent TE regions,
        and recurrent MoRE deliberately reuses modules across checkpoint
        boundaries whose backward order is reversed. A frame belongs to one
        checkpoint invocation, so each replay receives the exact pre-forward
        amax/scale tensors for its own surfaces without relying on TE's global
        FIFO recompute buffer.
        """

        frame = _ActivationCheckpointFrame(slots=[])
        return (
            self._activation_checkpoint_phase(
                frame=frame,
                recompute_phase=False,
            ),
            self._activation_checkpoint_phase(
                frame=frame,
                recompute_phase=True,
            ),
        )

    @contextlib.contextmanager
    def bf16_reference_context(self) -> Iterator[None]:
        """Execute TE modules without FP8 for the numerical parity canary."""

        self._force_bf16_depth += 1
        if self.device.type == "cuda":
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif self.device.type == "cpu":
            amp = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            amp = contextlib.nullcontext()
        try:
            with amp:
                if self._te is None:
                    yield
                    return
                fp8_context = getattr(self._te, "fp8_autocast", None)
                if fp8_context is None:
                    yield
                    return
                with fp8_context(enabled=False):
                    yield
        finally:
            self._force_bf16_depth -= 1

    def validate_execution(self) -> dict[str, Any]:
        """Run recurrent and chunked FP8 F/B before accepting the FP8 lane."""

        if not self.fp8_enabled:
            return {"ok": True, "profile": "bf16", "reason": self._fallback_reason}
        if self.device.type != "cuda":
            raise RuntimeError("FP8 execution validation requires a ROCm GPU")
        fp8_roles = (
            [
                role
                for role, dtype in self._measured_role_dtypes.items()
                if dtype == "fp8"
            ]
            if self._measured_role_dtypes is not None
            else [self.config.fp8_roles[0]]
        )
        if not fp8_roles:
            raise RuntimeError("FP8 execution validation has no FP8 role")
        validation_role = sorted(fp8_roles)[0]
        recurrent = self.linear(
            128,
            128,
            bias=False,
            role=validation_role,
        ).to(device=self.device, dtype=torch.bfloat16)
        head = self.linear(
            128,
            128,
            bias=False,
            role=validation_role,
        ).to(device=self.device, dtype=torch.bfloat16)
        plain_recurrent = self.linear(
            128,
            128,
            bias=False,
            role=validation_role,
        ).to(device=self.device, dtype=torch.bfloat16)
        plain_head = self.linear(
            128,
            128,
            bias=False,
            role=validation_role,
        ).to(device=self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            for target, source in zip(
                plain_recurrent.parameters(),
                recurrent.parameters(),
                strict=True,
            ):
                target.copy_(source)
            for target, source in zip(
                plain_head.parameters(),
                head.parameters(),
                strict=True,
            ):
                target.copy_(source)
        sample = torch.randn(
            65,
            128,
            device=self.device,
            dtype=torch.bfloat16,
        )

        reference_input = sample.detach().clone().requires_grad_(True)
        reference_output = reference_input
        for pass_index in range(3):
            with self.bf16_reference_context():
                update = recurrent(
                    reference_output * float(1.0 + 0.125 * (pass_index + 1))
                )
            reference_output = reference_output + 0.25 * update
        reference_chunk_losses: list[torch.Tensor] = []
        for chunk in reference_output.split((17, 15, 1, 32), dim=0):
            with self.bf16_reference_context():
                logits = head(chunk)
            reference_chunk_losses.append(logits.float().square().mean())
        reference_loss = torch.stack(reference_chunk_losses).mean()
        reference_loss.backward()
        if reference_input.grad is None:
            raise RuntimeError("BF16 recursive oracle produced no input gradient")
        reference_gradient = reference_input.grad.detach().clone()
        recurrent.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)

        plain_input = sample.detach().clone().requires_grad_(True)
        plain_output = plain_input
        for pass_index in range(3):
            with self.execution_context(module=plain_recurrent):
                update = plain_recurrent(
                    plain_output * float(1.0 + 0.125 * (pass_index + 1))
                )
            plain_output = plain_output + 0.25 * update
        plain_chunk_losses: list[torch.Tensor] = []
        for chunk in plain_output.split((17, 15, 1, 32), dim=0):
            with self.execution_context(module=plain_head):
                logits = plain_head(chunk)
            plain_chunk_losses.append(logits.float().square().mean())
        plain_loss = torch.stack(plain_chunk_losses).mean()
        plain_loss.backward()
        if plain_input.grad is None:
            raise RuntimeError("Plain FP8 recursive canary produced no input gradient")
        plain_gradient = plain_input.grad.detach().clone()
        plain_parameter_gradients = {
            f"{prefix}.{name}": parameter.grad.detach().clone()
            for prefix, module in (
                ("recurrent", plain_recurrent),
                ("head", plain_head),
            )
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }
        plain_recurrent_metadata = _fp8_metadata_digest(plain_recurrent)
        plain_head_metadata = _fp8_metadata_digest(plain_head)

        recurrent_amax: list[str] = []
        head_amax: list[str] = []
        fp8_input = sample.detach().clone().requires_grad_(True)
        output = fp8_input
        for pass_index in range(3):
            scale = float(1.0 + 0.125 * (pass_index + 1))

            def recurrent_step(
                values: torch.Tensor,
                _scale: float = scale,
            ) -> torch.Tensor:
                with self.execution_context(module=recurrent):
                    return recurrent(values * _scale)

            with set_checkpoint_early_stop(False):
                update = checkpoint(
                    recurrent_step,
                    output,
                    use_reentrant=False,
                    context_fn=self.activation_checkpoint_context_fn,
                )
            recurrent_amax.append(_fp8_metadata_digest(recurrent))
            output = output + 0.25 * update
        chunk_losses: list[torch.Tensor] = []
        for chunk in output.split((17, 15, 1, 32), dim=0):
            def head_step(values: torch.Tensor) -> torch.Tensor:
                with self.execution_context(module=head):
                    return head(values)

            with set_checkpoint_early_stop(False):
                logits = checkpoint(
                    head_step,
                    chunk,
                    use_reentrant=False,
                    context_fn=self.activation_checkpoint_context_fn,
                )
            head_amax.append(_fp8_metadata_digest(head))
            chunk_losses.append(logits.float().square().mean())
        loss = torch.stack(chunk_losses).mean()
        loss.backward()
        torch.cuda.synchronize(self.device)
        finite = bool(
            torch.isfinite(output).all().item()
            and torch.isfinite(loss).item()
            and fp8_input.grad is not None
            and torch.isfinite(fp8_input.grad).all().item()
            and all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for module in (recurrent, head)
                for parameter in module.parameters()
            )
        )
        if not finite:
            raise RuntimeError(
                "Transformer Engine recursive FP8 smoke test produced non-finite values"
            )
        output_relative_error = float(
            (output.float() - reference_output.float()).norm()
            .div(reference_output.float().norm().clamp_min(1.0e-12))
            .item()
        )
        loss_relative_error = abs(
            float(loss.detach().item()) - float(reference_loss.detach().item())
        ) / max(abs(float(reference_loss.detach().item())), 1.0e-12)
        gradient_relative_error = float(
            (fp8_input.grad.float() - reference_gradient.float()).norm()
            .div(reference_gradient.float().norm().clamp_min(1.0e-12))
            .item()
        )
        recompute_output_relative_error = float(
            (output.float() - plain_output.float()).norm()
            .div(plain_output.float().norm().clamp_min(1.0e-12))
            .item()
        )
        recompute_loss_relative_error = abs(
            float(loss.detach().item()) - float(plain_loss.detach().item())
        ) / max(abs(float(plain_loss.detach().item())), 1.0e-12)
        recompute_gradient_relative_error = float(
            (fp8_input.grad.float() - plain_gradient.float()).norm()
            .div(plain_gradient.float().norm().clamp_min(1.0e-12))
            .item()
        )
        checkpoint_parameter_gradients = {
            f"{prefix}.{name}": parameter.grad.detach()
            for prefix, module in (("recurrent", recurrent), ("head", head))
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }
        if set(checkpoint_parameter_gradients) != set(plain_parameter_gradients):
            raise RuntimeError(
                "FP8 checkpoint canary changed its parameter-gradient inventory"
            )
        recompute_parameter_gradient_relative_error = max(
            float(
                (
                    checkpoint_parameter_gradients[name].float()
                    - plain_parameter_gradients[name].float()
                )
                .norm()
                .div(
                    plain_parameter_gradients[name]
                    .float()
                    .norm()
                    .clamp_min(1.0e-12)
                )
                .item()
            )
            for name in checkpoint_parameter_gradients
        )
        if max(
            output_relative_error,
            loss_relative_error,
            gradient_relative_error,
        ) > 0.15:
            raise RuntimeError(
                "Transformer Engine recursive FP8 canary exceeded its BF16 "
                "composition-parity bound"
            )
        if max(
            recompute_output_relative_error,
            recompute_loss_relative_error,
            recompute_gradient_relative_error,
            recompute_parameter_gradient_relative_error,
        ) > 1.0e-4:
            raise RuntimeError(
                "Transformer Engine activation recompute differs from the "
                "non-checkpointed FP8 oracle"
            )
        if (
            _fp8_metadata_digest(recurrent) != plain_recurrent_metadata
            or _fp8_metadata_digest(head) != plain_head_metadata
        ):
            raise RuntimeError(
                "Transformer Engine activation recompute advanced delayed-scaling "
                "metadata beyond the non-checkpointed FP8 oracle"
            )
        if len(set(recurrent_amax)) < 2 or len(set(head_amax)) < 2:
            raise RuntimeError(
                "Transformer Engine delayed-scaling amax history did not advance "
                "across recursive passes and LM-head chunks"
            )
        return {
            "ok": True,
            "profile": "fp8",
            "output_dtype": str(output.dtype),
            "loss": float(loss.detach().item()),
            "transformer_engine_version": self._te_version,
            "validation_role": validation_role,
            "precision_role_plan_sha256": self.precision_role_plan_sha256,
            "recursive_passes": 3,
            "lm_head_chunks": 4,
            "ragged_rows": [65, 17, 15, 1, 32],
            "amax_history_advanced": True,
            "recurrent_amax_states": len(set(recurrent_amax)),
            "lm_head_amax_states": len(set(head_amax)),
            "output_relative_error_vs_bf16": output_relative_error,
            "loss_relative_error_vs_bf16": loss_relative_error,
            "gradient_relative_error_vs_bf16": gradient_relative_error,
            "recompute_output_relative_error_vs_plain_fp8": (
                recompute_output_relative_error
            ),
            "recompute_loss_relative_error_vs_plain_fp8": (
                recompute_loss_relative_error
            ),
            "recompute_input_gradient_relative_error_vs_plain_fp8": (
                recompute_gradient_relative_error
            ),
            "recompute_parameter_gradient_relative_error_vs_plain_fp8": (
                recompute_parameter_gradient_relative_error
            ),
            "recompute_metadata_matches_plain_fp8": True,
        }


def build_precision_policy(
    config: PrecisionConfig,
    *,
    profile: str,
    device: torch.device,
    production: bool,
    permit_fallback: bool | None = None,
    measured_role_dtypes: Mapping[str, str] | None = None,
    precision_role_plan_sha256: str | None = None,
) -> PrecisionPolicy:
    return PrecisionPolicy(
        config,
        requested_profile=profile,
        device=device,
        production=production,
        permit_fallback=permit_fallback,
        measured_role_dtypes=measured_role_dtypes,
        precision_role_plan_sha256=precision_role_plan_sha256,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_training_surface(
    operation: Any,
    *,
    device: torch.device,
    warmup_iterations: int,
    timed_iterations: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup_iterations):
        operation()
    _synchronize(device)
    samples: list[float] = []
    for _ in range(timed_iterations):
        started = time.perf_counter()
        operation()
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples)), samples


def benchmark_exact_precision_roles(
    config: Any,
    *,
    device: torch.device,
    warmup_iterations: int,
    timed_iterations: int,
    maximum_relative_error: float,
    minimum_fp8_speedup: float = 1.0,
    maximum_probe_rows: int = 2_048,
    maximum_activation_elements: int = 16_777_216,
) -> dict[str, Any]:
    """Measure real TE forward/backward independently for every exact role.

    A TE or exact-shape failure is a classified BF16 fallback, not a reason to
    discard safe FP8 execution on unrelated roles. BF16 itself is mandatory
    and any unsafe BF16 oracle fails the entire one-APU gate.
    """

    if (
        device.type != "cuda"
        or not torch.cuda.is_available()
        or not getattr(torch.version, "hip", None)
    ):
        raise RuntimeError("Exact-role precision probing requires a ROCm GPU")
    if warmup_iterations < 0 or timed_iterations <= 0:
        raise ValueError("Precision probe iteration counts are invalid")
    te, _recipe_module, te_version = _load_transformer_engine()
    fp8_policy: PrecisionPolicy | None = None
    te_error: str | None = None
    if te is not None:
        try:
            fp8_policy = PrecisionPolicy(
                config.precision,
                requested_profile="fp8",
                device=device,
                production=False,
                permit_fallback=False,
            )
        except BaseException as exc:
            te_error = f"{type(exc).__name__}: {exc}"
    else:
        te_error = "ROCm Transformer Engine is unavailable"

    measurements: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(exact_precision_role_specs(config)):
        rows = representative_probe_rows(
            spec,
            maximum_rows=maximum_probe_rows,
            maximum_activation_elements=maximum_activation_elements,
        )
        seed = 16_062_026 + index * 1_000_003
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        sample = torch.randn(
            rows,
            spec.in_features,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
            requires_grad=True,
        )
        reference = nn.Linear(
            spec.in_features,
            spec.out_features,
            bias=spec.bias,
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            reference.weight.normal_(
                mean=0.0,
                std=spec.in_features ** -0.5,
                generator=generator,
            )
            if reference.bias is not None:
                reference.bias.zero_()

        reference_output: torch.Tensor | None = None
        reference_loss_value = float("nan")

        def bf16_step() -> None:
            nonlocal reference_output, reference_loss_value
            reference.zero_grad(set_to_none=True)
            sample.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = reference(sample)
                loss = output.float().square().mean()
            loss.backward()
            reference_output = output.detach()
            reference_loss_value = float(loss.detach().item())

        bf16_median, bf16_samples = _timed_training_surface(
            bf16_step,
            device=device,
            warmup_iterations=warmup_iterations,
            timed_iterations=timed_iterations,
        )
        assert reference_output is not None
        bf16_finite = bool(
            torch.isfinite(reference_output).all().item()
            and math.isfinite(reference_loss_value)
            and sample.grad is not None
            and torch.isfinite(sample.grad).all().item()
            and all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in reference.parameters()
            )
        )
        if not bf16_finite:
            raise RuntimeError(
                f"Exact role {spec.role} produced a non-finite BF16 oracle"
            )
        train_flops = (
            6.0
            * rows
            * spec.in_features
            * spec.out_features
        )
        bf16_row: dict[str, Any] = {
            "ok": True,
            "finite_gradients": True,
            "median_seconds": bf16_median,
            "minimum_seconds": min(bf16_samples),
            "maximum_seconds": max(bf16_samples),
            "train_tflops": train_flops / bf16_median / 1.0e12,
            "loss": reference_loss_value,
            "output_dtype": str(reference_output.dtype),
        }
        fp8_row: dict[str, Any] = {
            "attempted": True,
            "ok": False,
            "finite_gradients": False,
            "error": te_error,
        }
        if fp8_policy is not None and te is not None:
            candidate: nn.Module | None = None
            fp8_sample: torch.Tensor | None = None
            try:
                candidate = te.Linear(
                    spec.in_features,
                    spec.out_features,
                    bias=spec.bias,
                ).to(device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    candidate.weight.copy_(reference.weight)
                    candidate_bias = getattr(candidate, "bias", None)
                    if candidate_bias is not None and reference.bias is not None:
                        candidate_bias.copy_(reference.bias)
                fp8_sample = sample.detach().clone().requires_grad_(True)
                candidate_output: torch.Tensor | None = None
                candidate_loss_value = float("nan")

                def fp8_step() -> None:
                    nonlocal candidate_output, candidate_loss_value
                    assert candidate is not None and fp8_sample is not None
                    candidate.zero_grad(set_to_none=True)
                    fp8_sample.grad = None
                    with fp8_policy.execution_context():
                        output = candidate(fp8_sample)
                        loss = output.float().square().mean()
                    loss.backward()
                    candidate_output = output.detach()
                    candidate_loss_value = float(loss.detach().item())

                fp8_median, fp8_samples = _timed_training_surface(
                    fp8_step,
                    device=device,
                    warmup_iterations=warmup_iterations,
                    timed_iterations=timed_iterations,
                )
                assert candidate_output is not None
                finite = bool(
                    torch.isfinite(candidate_output).all().item()
                    and math.isfinite(candidate_loss_value)
                    and fp8_sample.grad is not None
                    and torch.isfinite(fp8_sample.grad).all().item()
                    and all(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all().item()
                        for parameter in candidate.parameters()
                    )
                )
                loss_relative_error = abs(
                    candidate_loss_value - reference_loss_value
                ) / max(abs(reference_loss_value), 1.0e-12)
                output_relative_l2_error = float(
                    (
                        candidate_output.float() - reference_output.float()
                    ).norm()
                    .div(reference_output.float().norm().clamp_min(1.0e-12))
                    .item()
                )
                fp8_row = {
                    "attempted": True,
                    "ok": finite,
                    "finite_gradients": finite,
                    "median_seconds": fp8_median,
                    "minimum_seconds": min(fp8_samples),
                    "maximum_seconds": max(fp8_samples),
                    "train_tflops": train_flops / fp8_median / 1.0e12,
                    "loss": candidate_loss_value,
                    "loss_relative_error_vs_bf16": loss_relative_error,
                    "output_relative_l2_error_vs_bf16": output_relative_l2_error,
                    "speedup_vs_bf16": bf16_median / fp8_median,
                    "output_dtype": str(candidate_output.dtype),
                    "error": None if finite else "non-finite FP8 output or gradient",
                }
            except BaseException as exc:
                fp8_row = {
                    "attempted": True,
                    "ok": False,
                    "finite_gradients": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                del candidate, fp8_sample
        measurements[spec.role] = {
            "shape": {
                "rows": rows,
                "in_features": spec.in_features,
                "out_features": spec.out_features,
                "bias": spec.bias,
            },
            "bf16": bf16_row,
            "fp8": fp8_row,
        }
        del reference, sample, reference_output
        torch.cuda.empty_cache()

    plan = build_precision_role_plan(
        config,
        measurements,
        maximum_relative_error=maximum_relative_error,
        minimum_fp8_speedup=minimum_fp8_speedup,
    )
    inventory = build_precision_role_inventory(config)
    return {
        "ok": True,
        "transformer_engine_version": te_version,
        "measurements": measurements,
        "precision_role_inventory": inventory,
        "precision_role_plan": plan,
        "precision_role_plan_sha256": plan["plan_sha256"],
        "precision_role_inventory_sha256": inventory["inventory_sha256"],
    }
