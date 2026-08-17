from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import torch
from torch import nn
from torch.optim import Optimizer


def _nvtx_range(name: str):
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        return torch.cuda.nvtx.range(name)
    return nullcontext()


@dataclass
class OptimizerGroupSummary:
    name: str
    optimizer: str
    tensor_count: int
    param_count: int
    sample_names: list[str]


@dataclass
class OptimizerBuildSummary:
    name: str
    total_params: int
    muon_params: int
    adamw_params: int
    routed_experts_muon: bool
    adamw_impl: str
    master_weights: bool
    groups: list[OptimizerGroupSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _zeropower_via_newton_schulz5(update: torch.Tensor, *, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a matrix, or a stack of them, by five Newton-Schulz steps.

    A stacked expert bank presents its projections as one
    ``[experts, out, in]`` parameter rather than one 2D parameter per expert.
    Muon's semantics are per matrix either way -- each expert is orthogonalized
    against itself and no other -- so the leading dimensions are carried as a
    batch and every reduction is taken per matrix. Handling both shapes in one
    function is deliberate: a separate batched implementation would be free to
    drift away from the 2D one it is supposed to reproduce.
    """

    if update.ndim < 2:
        raise ValueError("Muon updates require at least a 2D tensor.")
    if steps <= 0:
        return update
    x = update.float()
    norm = x.flatten(start_dim=-2).norm(dim=-1)[..., None, None]
    valid_norm = torch.isfinite(norm) & (norm > 0)
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.transpose(-2, -1)
    safe_norm = torch.where(valid_norm, norm.clamp_min(eps), torch.ones_like(norm))
    x = torch.where(valid_norm, x / safe_norm, torch.zeros_like(x))
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.transpose(-2, -1)
        x = (a * x) + ((b * xx_t + c * (xx_t @ xx_t)) @ x)
    if transpose:
        x = x.transpose(-2, -1)
    return x


def _muon_update_scale(rows: int, cols: int, mode: str) -> float:
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(float(max(rows, cols, 1)))
    if mode == "original":
        return math.sqrt(max(1.0, float(rows) / float(max(cols, 1))))
    raise ValueError("muon_scale_mode must be one of: original, match_rms_adamw.")


class MuonAdamWHybrid(Optimizer):
    """Single optimizer object with AdamW groups and Muon matrix groups."""

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        muon_beta: float,
        muon_ns_steps: int,
        muon_nesterov: bool,
        muon_scale_mode: str = "match_rms_adamw",
        adamw_impl: str = "foreach",
        master_weights: bool = False,
    ) -> None:
        if muon_scale_mode not in {"original", "match_rms_adamw"}:
            raise ValueError("muon_scale_mode must be one of: original, match_rms_adamw.")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "muon_beta": muon_beta,
            "muon_ns_steps": muon_ns_steps,
            "muon_nesterov": muon_nesterov,
            "muon_scale_mode": muon_scale_mode,
            "adamw_impl": adamw_impl,
            "master_weights": master_weights,
        }
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mode = group.get("optimizer", "adamw")
            if mode == "muon":
                with _nvtx_range("optimizer_muon"):
                    self._step_muon_group(group)
            elif mode == "adamw":
                with _nvtx_range("optimizer_adamw"):
                    self._step_adamw_group(group)
            else:
                raise RuntimeError(f"Unknown optimizer group mode: {mode}")
        return loss

    @torch.no_grad()
    def sync_master_params_from_model(self, params: Iterable[nn.Parameter] | None = None) -> None:
        """Reflect external in-place parameter edits, such as QK clip, into FP32 masters."""

        wanted = None if params is None else {id(param) for param in params}
        for group in self.param_groups:
            if not bool(group.get("master_weights", False)):
                continue
            for param in group["params"]:
                if wanted is not None and id(param) not in wanted:
                    continue
                state = self.state.get(param, {})
                master_param = state.get("master_param")
                if master_param is not None:
                    master_param.copy_(param.detach().float())

    def _step_adamw_group(self, group: dict[str, Any]) -> None:
        params = [param for param in group["params"] if param.grad is not None]
        has_xla_param = any(str(param.device).startswith("xla") for param in params)
        use_master_weights = bool(group.get("master_weights", False))
        if (
            group.get("adamw_impl", "foreach") == "foreach"
            and hasattr(torch, "_foreach_mul_")
            and not has_xla_param
            and not use_master_weights
        ):
            self._step_adamw_group_foreach(group)
            return
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonAdamWHybrid does not support sparse gradients.")
            grad_f = grad.float()
            state = self.state[param]
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=torch.float32, device=param.device)
                    if str(param.device).startswith("xla")
                    else 0
                )
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
                if use_master_weights:
                    state["master_param"] = param.detach().float().clone()
            elif use_master_weights and "master_param" not in state:
                state["master_param"] = param.detach().float().clone()
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            param_data = state["master_param"] if use_master_weights else param
            if weight_decay != 0:
                param_data.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad_f, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1.0 - beta2)
            update = exp_avg / exp_avg_sq.sqrt().add_(eps)
            step_state = state["step"]
            if isinstance(step_state, torch.Tensor):
                step_state.add_(1.0)
                beta1_t = torch.tensor(beta1, dtype=torch.float32, device=param.device)
                beta2_t = torch.tensor(beta2, dtype=torch.float32, device=param.device)
                lr_t = torch.tensor(lr, dtype=torch.float32, device=param.device)
                bias_correction1 = 1.0 - torch.pow(beta1_t, step_state)
                bias_correction2 = 1.0 - torch.pow(beta2_t, step_state)
                step_size = lr_t * torch.sqrt(bias_correction2) / bias_correction1.clamp_min(1e-16)
                param_data.add_(update * (-step_size))
            else:
                state["step"] = int(step_state) + 1
                step = int(state["step"])
                bias_correction1 = 1.0 - (beta1**step)
                bias_correction2 = 1.0 - (beta2**step)
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                param_data.add_(update, alpha=-step_size)
            if use_master_weights:
                param.copy_(param_data.to(dtype=param.dtype))

    def _step_adamw_group_foreach(self, group: dict[str, Any]) -> None:
        beta1, beta2 = group["betas"]
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        exp_avgs: list[torch.Tensor] = []
        exp_avg_sqs: list[torch.Tensor] = []
        steps: list[int] = []
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonAdamWHybrid does not support sparse gradients.")
            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["step"] += 1
            params.append(param)
            grads.append(grad.detach().float())
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            steps.append(int(state["step"]))

        if not params:
            return
        if weight_decay != 0:
            torch._foreach_mul_(params, 1.0 - lr * weight_decay)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        denoms = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(denoms, eps)
        updates = torch._foreach_div(exp_avgs, denoms)
        for param, update, step in zip(params, updates, steps, strict=True):
            bias_correction1 = 1.0 - (beta1**step)
            bias_correction2 = 1.0 - (beta2**step)
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            param.add_(update.to(dtype=param.dtype), alpha=-step_size)

    def _step_muon_group(self, group: dict[str, Any]) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta = group["muon_beta"]
        ns_steps = int(group["muon_ns_steps"])
        nesterov = bool(group["muon_nesterov"])
        scale_mode = str(group.get("muon_scale_mode", "match_rms_adamw"))
        use_master_weights = bool(group.get("master_weights", False))
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("MuonAdamWHybrid does not support sparse gradients.")
            if grad.ndim not in (2, 3):
                raise RuntimeError(
                    "Muon groups may only contain matrix parameters, or one "
                    "stack of them."
                )
            grad_f = grad.float()
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param, dtype=torch.float32)
                if use_master_weights:
                    state["master_param"] = param.detach().float().clone()
            elif use_master_weights and "master_param" not in state:
                state["master_param"] = param.detach().float().clone()
            momentum = state["momentum_buffer"]
            momentum.mul_(beta).add_(grad_f)
            update = grad_f.add(momentum, alpha=beta) if nesterov else momentum
            update = _zeropower_via_newton_schulz5(update, steps=ns_steps)
            rows, cols = int(param.shape[-2]), int(param.shape[-1])
            update_scale = _muon_update_scale(rows, cols, scale_mode)
            param_data = state["master_param"] if use_master_weights else param
            if weight_decay != 0:
                param_data.mul_(1.0 - lr * weight_decay)
            param_data.add_(update, alpha=-(lr * update_scale))
            if use_master_weights:
                param.copy_(param_data.to(dtype=param.dtype))


def _is_bias_or_vector(name: str, param: nn.Parameter) -> bool:
    lowered = name.lower()
    return param.ndim < 2 or lowered.endswith("bias") or ".bias" in lowered or "_bias" in lowered


def _is_norm(name: str) -> bool:
    lowered = name.lower()
    return "norm" in lowered or "layernorm" in lowered or "rmsnorm" in lowered


def _classify_param(
    name: str,
    param: nn.Parameter,
    *,
    include_routed_experts: bool,
) -> tuple[str, str, bool]:
    lowered = name.lower()
    if _is_bias_or_vector(name, param):
        return "adamw", "bias_vector_or_scalar", False
    if "embed_tokens" in lowered or "lm_head" in lowered:
        return "adamw", "embedding_or_lm_head", True
    if _is_norm(lowered):
        return "adamw", "normalization", False
    if "router" in lowered or "expert_embeddings" in lowered:
        return "adamw", "router_or_gate_control", True
    if "latent_proj" in lowered:
        return "adamw", "latent_moe_router_projection", True
    if ".moe.latent_down." in lowered or ".moe.latent_up." in lowered:
        return "muon", "metis16_latent_projection", True
    if ".moe.local_experts." in lowered and (
        ".gate_up." in lowered or ".down." in lowered
    ):
        if include_routed_experts:
            return "muon", "metis16_routed_expert_projection", True
        return "adamw", "metis16_routed_expert_waiting_ablation", True
    if ".moe.shared_expert." in lowered and (
        ".gate_up." in lowered or ".down." in lowered
    ):
        return "muon", "metis16_shared_expert_projection", True
    if (
        ".mixer.impl.qkv." in lowered
        or ".mixer.impl.out." in lowered
        or ".mixer.impl.pass_lora_" in lowered
        or ".mixer.impl.mixer.in_proj." in lowered
        or ".mixer.impl.mixer.out_proj." in lowered
    ):
        return "muon", "metis16_mixer_projection", True
    if (
        lowered.startswith("depth_memory.")
        or lowered.startswith("ngram_memory.projection.")
        or ".mixer_connection.controller." in lowered
        or ".moe_connection.controller." in lowered
    ) and param.ndim == 2:
        return "muon", "metis16_control_or_memory_projection", True
    if ".attn." in lowered and ("qkv_proj" in lowered or "o_proj" in lowered):
        return "muon", "tpu_attention_projection", True
    if ".moe.down_proj" in lowered or ".moe.up_proj" in lowered:
        return "muon", "tpu_latent_moe_payload_projection", True
    if ".moe.shared." in lowered and (".up." in lowered or ".down." in lowered):
        return "muon", "tpu_shared_expert_projection", True
    if ".moe.dense_ffn." in lowered and (".up." in lowered or ".down." in lowered):
        return "muon", "tpu_dense_ffn_projection", True
    if ".moe.experts." in lowered and (".up." in lowered or ".down." in lowered):
        if include_routed_experts:
            return "muon", "tpu_routed_expert_projection_ablation", True
        return "adamw", "tpu_routed_expert_projection_waiting_ablation", True
    if "routed_down_proj" in lowered or "routed_up_proj" in lowered:
        return "muon", "latent_moe_payload_projection", True
    if ".self_attn." in lowered and (
        "qkv_proj" in lowered or "o_proj" in lowered
    ):
        return "muon", "attention_projection", True
    if ".mlp.shared." in lowered and (
        "gate_up_proj" in lowered or "down_proj" in lowered
    ):
        return "muon", "shared_expert_projection", True
    if ".mlp.grouped_experts." in lowered and (
        "gate_up_proj" in lowered or "down_proj" in lowered
    ):
        if include_routed_experts:
            return "muon", "routed_grouped_expert_projection_ablation", True
        return "adamw", "routed_grouped_expert_projection_waiting_ablation", True
    if ".mlp.experts." in lowered and (
        "gate_up_proj" in lowered or "down_proj" in lowered
    ):
        if include_routed_experts:
            return "muon", "routed_expert_projection_ablation", True
        return "adamw", "routed_expert_projection_waiting_ablation", True
    if ".mlp." in lowered and ("gate_up_proj" in lowered or "down_proj" in lowered):
        return "muon", "dense_mlp_projection", True
    return "adamw", "conservative_default", param.ndim >= 2


def build_muon_adamw_optimizer(
    model: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float = 1e-8,
    weight_decay: float,
    muon_beta: float = 0.95,
    muon_ns_steps: int = 5,
    muon_lr_scale: float = 1.0,
    muon_nesterov: bool = True,
    muon_scale_mode: str = "match_rms_adamw",
    include_routed_experts: bool = False,
    adamw_impl: str = "foreach",
    master_weights: bool = False,
) -> tuple[MuonAdamWHybrid, OptimizerBuildSummary]:
    if muon_scale_mode not in {"original", "match_rms_adamw"}:
        raise ValueError("muon_scale_mode must be one of: original, match_rms_adamw.")
    buckets: dict[str, dict[str, Any]] = {
        "adamw_decay": {"params": [], "names": [], "optimizer": "adamw", "weight_decay": weight_decay},
        "adamw_no_decay": {"params": [], "names": [], "optimizer": "adamw", "weight_decay": 0.0},
        "muon": {"params": [], "names": [], "optimizer": "muon", "weight_decay": weight_decay},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mode, _reason, use_decay = _classify_param(
            name,
            param,
            include_routed_experts=include_routed_experts,
        )
        if mode == "muon":
            # A stacked expert bank is one parameter holding every expert's
            # matrix; Muon orthogonalizes each of them separately, so a 3D
            # parameter is a batch of matrices rather than a shape error.
            if param.ndim not in (2, 3):
                raise RuntimeError(
                    f"Muon-classified parameter is not a matrix or a stack of "
                    f"them: {name} {tuple(param.shape)}"
                )
            bucket_name = "muon"
        elif use_decay:
            bucket_name = "adamw_decay"
        else:
            bucket_name = "adamw_no_decay"
        buckets[bucket_name]["params"].append(param)
        buckets[bucket_name]["names"].append(name)

    param_groups: list[dict[str, Any]] = []
    group_summaries: list[OptimizerGroupSummary] = []
    for bucket_name, bucket in buckets.items():
        params = bucket["params"]
        if not params:
            continue
        mode = bucket["optimizer"]
        group_lr = lr * (muon_lr_scale if mode == "muon" else 1.0)
        group = {
            "params": params,
            "lr": group_lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": bucket["weight_decay"],
            "optimizer": mode,
            "muon_beta": muon_beta,
            "muon_ns_steps": muon_ns_steps,
            "muon_nesterov": muon_nesterov,
            "muon_scale_mode": muon_scale_mode,
            "adamw_impl": adamw_impl,
            "master_weights": master_weights,
        }
        param_groups.append(group)
        group_summaries.append(
            OptimizerGroupSummary(
                name=bucket_name,
                optimizer=mode,
                tensor_count=len(params),
                param_count=sum(param.numel() for param in params),
                sample_names=bucket["names"][:8],
            )
        )

    optimizer = MuonAdamWHybrid(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        muon_beta=muon_beta,
        muon_ns_steps=muon_ns_steps,
        muon_nesterov=muon_nesterov,
        muon_scale_mode=muon_scale_mode,
        adamw_impl=adamw_impl,
        master_weights=master_weights,
    )
    muon_params = sum(group.param_count for group in group_summaries if group.optimizer == "muon")
    adamw_params = sum(group.param_count for group in group_summaries if group.optimizer == "adamw")
    summary = OptimizerBuildSummary(
        name="muon_adamw",
        total_params=muon_params + adamw_params,
        muon_params=muon_params,
        adamw_params=adamw_params,
        routed_experts_muon=include_routed_experts,
        adamw_impl=adamw_impl,
        master_weights=master_weights,
        groups=group_summaries,
    )
    return optimizer, summary


def build_xla_stable_adamw_optimizer(
    model: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float = 1e-8,
    weight_decay: float,
    adamw_impl: str = "loop",
    master_weights: bool = False,
) -> tuple[MuonAdamWHybrid, OptimizerBuildSummary]:
    buckets: dict[str, dict[str, Any]] = {
        "adamw_decay": {"params": [], "names": [], "weight_decay": weight_decay},
        "adamw_no_decay": {"params": [], "names": [], "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_decay = param.ndim >= 2 and not _is_norm(name) and not _is_bias_or_vector(name, param)
        bucket = buckets["adamw_decay" if use_decay else "adamw_no_decay"]
        bucket["params"].append(param)
        bucket["names"].append(name)

    param_groups: list[dict[str, Any]] = []
    group_summaries: list[OptimizerGroupSummary] = []
    for bucket_name, bucket in buckets.items():
        params = bucket["params"]
        if not params:
            continue
        param_groups.append(
            {
                "params": params,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": bucket["weight_decay"],
                "optimizer": "adamw",
                "muon_beta": 0.95,
                "muon_ns_steps": 0,
                "muon_nesterov": False,
                "muon_scale_mode": "match_rms_adamw",
                "adamw_impl": adamw_impl,
                "master_weights": master_weights,
            }
        )
        group_summaries.append(
            OptimizerGroupSummary(
                name=bucket_name,
                optimizer="adamw",
                tensor_count=len(params),
                param_count=sum(param.numel() for param in params),
                sample_names=bucket["names"][:8],
            )
        )

    optimizer = MuonAdamWHybrid(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        muon_beta=0.95,
        muon_ns_steps=0,
        muon_nesterov=False,
        muon_scale_mode="match_rms_adamw",
        adamw_impl=adamw_impl,
        master_weights=master_weights,
    )
    total_params = sum(group.param_count for group in group_summaries)
    summary = OptimizerBuildSummary(
        name="xla_stable_adamw",
        total_params=total_params,
        muon_params=0,
        adamw_params=total_params,
        routed_experts_muon=False,
        adamw_impl=adamw_impl,
        master_weights=master_weights,
        groups=group_summaries,
    )
    return optimizer, summary


def build_optimizer_from_args(
    model: nn.Module,
    args,
    optimizer_manifest: dict[str, object] | None = None,
) -> tuple[Optimizer, OptimizerBuildSummary | None]:
    optimizer_manifest = optimizer_manifest or {}
    optimizer_name = str(getattr(args, "optimizer", None) or optimizer_manifest.get("name", "adamw")).lower()
    master_weights = bool(
        getattr(args, "optimizer_master_weights", False)
        or optimizer_manifest.get("master_weights", False)
        or optimizer_manifest.get("fp32_master_weights", False)
    )
    if optimizer_name in {"muon-adamw", "muon_adamw", "hybrid_muon_adamw"}:
        include_routed_experts = bool(
            getattr(args, "muon_include_routed_experts", False)
            or optimizer_manifest.get("include_routed_experts", False)
            or optimizer_manifest.get("routed_experts_muon", False)
        )
        muon_beta_arg = getattr(args, "muon_beta", None)
        muon_ns_steps_arg = getattr(args, "muon_ns_steps", None)
        muon_lr_scale_arg = getattr(args, "muon_lr_scale", None)
        muon_scale_mode = str(
            getattr(args, "muon_scale_mode", None)
            or optimizer_manifest.get("muon_scale_mode", "match_rms_adamw")
        ).lower()
        adamw_impl = str(
            getattr(args, "hybrid_adamw_impl", None)
            or optimizer_manifest.get("hybrid_adamw_impl", "foreach")
        ).lower()
        if bool(getattr(args, "fused_adamw", False)):
            adamw_impl = "foreach"
        if adamw_impl not in {"loop", "foreach"}:
            raise ValueError("hybrid_adamw_impl must be one of: loop, foreach.")
        return build_muon_adamw_optimizer(
            model,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(optimizer_manifest.get("adamw_eps", 1e-8)),
            weight_decay=float(args.weight_decay),
            muon_beta=float(muon_beta_arg if muon_beta_arg is not None else optimizer_manifest.get("muon_beta", 0.95)),
            muon_ns_steps=int(
                muon_ns_steps_arg if muon_ns_steps_arg is not None else optimizer_manifest.get("muon_ns_steps", 5)
            ),
            muon_lr_scale=float(
                muon_lr_scale_arg if muon_lr_scale_arg is not None else optimizer_manifest.get("muon_lr_scale", 1.0)
            ),
            muon_nesterov=bool(optimizer_manifest.get("muon_nesterov", True)),
            muon_scale_mode=muon_scale_mode,
            include_routed_experts=include_routed_experts,
            adamw_impl=adamw_impl,
            master_weights=master_weights,
        )
    if optimizer_name != "adamw":
        raise ValueError("optimizer must be one of: adamw, muon_adamw.")
    if bool(getattr(args, "xla_stable_adamw", False)) or master_weights:
        adamw_impl = str(getattr(args, "hybrid_adamw_impl", "loop") or "loop").lower()
        if adamw_impl not in {"loop", "foreach"}:
            raise ValueError("hybrid_adamw_impl must be one of: loop, foreach.")
        return build_xla_stable_adamw_optimizer(
            model,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(optimizer_manifest.get("adamw_eps", 1e-8)),
            weight_decay=float(args.weight_decay),
            adamw_impl=adamw_impl,
            master_weights=master_weights,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(float(args.beta1), float(args.beta2)),
        weight_decay=float(args.weight_decay),
        fused=bool(getattr(args, "fused_adamw", False) and torch.cuda.is_available()),
    )
    return optimizer, None
