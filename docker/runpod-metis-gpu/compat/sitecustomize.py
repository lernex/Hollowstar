from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


def _patch_checkpoint_staging() -> None:
    try:
        import torch.distributed.checkpoint.staging as staging
    except Exception:
        return

    if not hasattr(staging, "StagingOptions"):
        @dataclass
        class StagingOptions:
            use_async_staging: bool = True
            use_pinned_memory: bool = True
            use_shared_memory: bool = True
            use_non_blocking_copy: bool = True

        staging.StagingOptions = StagingOptions

    if not hasattr(staging, "DefaultStager"):
        class DefaultStager(staging.BlockingAsyncStager):
            def __init__(self, options: object | None = None) -> None:
                super().__init__()
                self.options = options

        staging.DefaultStager = DefaultStager


def _patch_state_dict_saver() -> None:
    try:
        import torch.distributed.checkpoint.state_dict_saver as saver
    except Exception:
        return

    if not hasattr(saver, "AsyncSaveResponse"):
        class AsyncSaveResponse(SimpleNamespace):
            pass

        saver.AsyncSaveResponse = AsyncSaveResponse


def _patch_context_parallel_attention() -> None:
    try:
        import torch.distributed.tensor.experimental._attention as attention
    except Exception:
        return

    if not hasattr(attention, "_context_parallel_shard"):
        def _context_parallel_shard(
            mesh: object,
            buffers: object,
            seq_dims: object,
            load_balancer: object | None = None,
        ) -> object:
            del mesh, seq_dims, load_balancer
            return buffers

        attention._context_parallel_shard = _context_parallel_shard

    if not hasattr(attention, "_enable_context_parallel_dispatcher"):
        def _enable_context_parallel_dispatcher() -> None:
            return None

        attention._enable_context_parallel_dispatcher = (
            _enable_context_parallel_dispatcher
        )

    if not hasattr(attention, "_ContextParallel"):
        class _ContextParallel:
            class AttentionType:
                FLEX = "flex"
                SDPA = "sdpa"

            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        attention._ContextParallel = _ContextParallel

    if not hasattr(attention, "_HeadTailLoadBalancer"):
        class _HeadTailLoadBalancer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        attention._HeadTailLoadBalancer = _HeadTailLoadBalancer

    if not hasattr(attention, "_PTRRLoadBalancer"):
        class _PTRRLoadBalancer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        attention._PTRRLoadBalancer = _PTRRLoadBalancer


def _patch_torch_nn_attention() -> None:
    try:
        import torch.nn.attention as attention
    except Exception:
        return

    if not hasattr(attention, "activate_flash_attention_impl"):
        def activate_flash_attention_impl(name: str) -> None:
            setattr(attention, "_metis_flash_attention_impl", name)

        attention.activate_flash_attention_impl = activate_flash_attention_impl

    if not hasattr(attention, "current_flash_attention_impl"):
        def current_flash_attention_impl() -> str | None:
            return getattr(attention, "_metis_flash_attention_impl", None)

        attention.current_flash_attention_impl = current_flash_attention_impl


def _patch_flex_attention() -> None:
    try:
        import torch.nn.attention.flex_attention as flex_attention
    except Exception:
        return

    if not hasattr(flex_attention, "AuxRequest"):
        @dataclass
        class AuxRequest:
            lse: bool = False

        flex_attention.AuxRequest = AuxRequest


_patch_checkpoint_staging()
_patch_state_dict_saver()
_patch_context_parallel_attention()
_patch_torch_nn_attention()
_patch_flex_attention()
