"""Training contracts and orchestration for the Metis model family.

The public post-training helpers are loaded lazily so login-node operator
commands can inspect or launch a campaign without importing the compute-only
PyTorch runtime.
"""

from importlib import import_module
from typing import Any


__all__ = [
    "PipelineContractError",
    "PostTrainingOrchestrator",
    "difficulty_adaptive_length_budget",
    "difficulty_adaptive_length_reward",
    "evaluate_metric_gate",
    "gated_code_efficiency_reward",
    "gspo_loss",
    "gspo_token_loss",
    "load_pipeline",
    "masked_causal_cross_entropy",
    "strict_on_policy_filter",
    "validate_pipeline",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(".posttraining", __name__), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
