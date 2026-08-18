"""Context parallelism and layer-level activation recompute.

The multi-rank cases run real gloo process groups in spawned workers rather
than a mocked collective, because the failures that matter here are collective
failures: a rank that skips a backward reduce-scatter, or one that leaves the
pass loop a pass early, produces a hang rather than a wrong number, and only a
real group reproduces it.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from metis_training.context_parallel import (
    ContextParallelContext,
    build_context_parallel_attention_layout,
    conv_left_halo,
    left_halo,
    mamba_incoming_state,
    mamba_shard_summary,
    packed_segment_keys,
)
from metis_training.model import (
    CurriculumState,
    Metis16ForCausalLM,
    MetisProcessGroups,
    _packed_document_metadata,
)
from metis_training.model_config import Metis16Config


BATCH = 1
LENGTH = 16
SEED = 7


# ---------------------------------------------------------------------------
# Single-process: configuration, recompute parity, closed-form SSD summary
# ---------------------------------------------------------------------------


def _curriculum() -> CurriculumState:
    return CurriculumState(
        routed_k_mode="fixed",
        fixed_routed_k=2,
        stochastic_routing=False,
        target_mean_depth=2.0,
        target_mean_routed_k=2.0,
    )


def _batch(config: Metis16Config):
    generator = torch.Generator().manual_seed(16_062_026)
    input_ids = torch.randint(0, config.vocab_size, (BATCH, LENGTH), generator=generator)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    reset_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    reset_mask[:, 0] = True
    reset_mask[:, 5] = True  # inside a shard
    reset_mask[:, 8] = True  # exactly on a CP=2 and CP=4 shard edge
    positions = torch.arange(LENGTH).view(1, LENGTH).repeat(BATCH, 1)
    forced_depths = (positions % config.max_passes) + 1
    return input_ids, labels, reset_mask, forced_depths


def _run_single(policy: str):
    torch.manual_seed(SEED)
    config = replace(Metis16Config.tiny_for_tests(), activation_recompute_policy=policy)
    model = Metis16ForCausalLM(config)
    model.train()
    input_ids, labels, reset_mask, forced_depths = _batch(config)
    output = model(
        input_ids,
        labels=labels,
        reset_mask=reset_mask,
        curriculum=_curriculum(),
        force_depth=forced_depths,
    )
    output.loss.backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not parameter.grad.is_sparse
    }
    return output, gradients


@pytest.mark.parametrize("policy", ["none", "pass", "layer"])
def test_activation_recompute_policies_are_accepted(policy: str) -> None:
    config = replace(Metis16Config.tiny_for_tests(), activation_recompute_policy=policy)
    config._validate_tiny()
    assert config.activation_recompute_policy == policy


def test_unknown_activation_recompute_policy_is_rejected() -> None:
    config = replace(Metis16Config.tiny_for_tests(), activation_recompute_policy="block")
    with pytest.raises(ValueError, match="none, pass, or layer"):
        config._validate_tiny()


def test_context_parallelism_requires_activation_recompute() -> None:
    # Sharding the sequence and then keeping every activation spends the
    # communication without buying the memory back.
    config = replace(
        Metis16Config.tiny_for_tests(),
        context_parallel_size=2,
        activation_recompute_policy="none",
    )
    with pytest.raises(ValueError, match="requires activation"):
        config._validate_tiny()


def test_context_parallel_size_must_divide_the_context_length() -> None:
    config = replace(
        Metis16Config.tiny_for_tests(),
        context_parallel_size=5,
        activation_recompute_policy="layer",
    )
    with pytest.raises(ValueError, match="not divisible"):
        config._validate_tiny()


def test_context_parallel_group_and_manifest_must_agree() -> None:
    config = replace(Metis16Config.tiny_for_tests(), context_parallel_size=1)
    with pytest.raises(RuntimeError, match="context_parallel_size is 1"):
        Metis16ForCausalLM(config, process_groups=MetisProcessGroups(context=object()))


def test_layer_recompute_matches_pass_recompute_exactly() -> None:
    # Both policies replay the same forward; only the granularity differs, so
    # any divergence here is a bug in the layer checkpoint boundary rather than
    # a rounding difference.
    pass_output, pass_gradients = _run_single("pass")
    layer_output, layer_gradients = _run_single("layer")
    assert torch.equal(pass_output.loss, layer_output.loss)
    assert set(pass_gradients) == set(layer_gradients)
    for name, expected in pass_gradients.items():
        assert torch.equal(expected, layer_gradients[name]), name


def test_layer_recompute_is_reported_in_telemetry() -> None:
    for policy, expected in (("none", 0), ("pass", 1), ("layer", 1)):
        output, _ = _run_single(policy)
        assert int(output.telemetry["activation_recompute_enabled"].item()) == expected


def test_shard_summary_reproduces_an_explicit_ssd_scan() -> None:
    """The closed form must equal the recurrence it replaces, resets included."""

    torch.manual_seed(3)
    batch, seq, heads, head_dim, d_state = 2, 12, 3, 4, 5
    x = torch.randn(batch, seq, heads, head_dim, dtype=torch.float64)
    b_matrix = torch.randn(batch, seq, heads, d_state, dtype=torch.float64)
    delta = torch.rand(batch, seq, heads, dtype=torch.float64) * 0.5 + 0.05
    a_log = torch.randn(heads, dtype=torch.float64) * 0.3
    reset = torch.zeros(batch, seq, dtype=torch.bool)
    reset[:, 0] = True
    reset[:, 7] = True

    decay, state = mamba_shard_summary(x, b_matrix, delta, a_log, reset_mask=reset)

    rate = -torch.exp(a_log.double())
    scanned = torch.zeros(batch, heads, head_dim, d_state, dtype=torch.float64)
    carried = torch.ones(batch, heads, dtype=torch.float64)
    for step in range(seq):
        keep = (~reset[:, step]).double().view(batch, 1, 1, 1)
        scanned = scanned * keep
        carried = carried * keep.view(batch, 1)
        transition = torch.exp(delta[:, step][:, :, None, None] * rate[None, :, None, None])
        scanned = scanned * transition + (
            delta[:, step][:, :, None, None]
            * x[:, step][:, :, :, None]
            * b_matrix[:, step][:, :, None, :]
        )
        carried = carried * transition[:, :, 0, 0]
    assert torch.allclose(state, scanned, atol=1e-12)
    assert torch.allclose(decay, carried, atol=1e-12)


def test_shard_summary_zeroes_the_carry_after_any_reset() -> None:
    # A reset anywhere in the shard means no incoming state survives to its end.
    torch.manual_seed(5)
    x = torch.randn(1, 6, 2, 3, dtype=torch.float64)
    b_matrix = torch.randn(1, 6, 2, 4, dtype=torch.float64)
    delta = torch.rand(1, 6, 2, dtype=torch.float64) + 0.1
    a_log = torch.zeros(2, dtype=torch.float64)
    reset = torch.zeros(1, 6, dtype=torch.bool)
    decay_open, _ = mamba_shard_summary(x, b_matrix, delta, a_log, reset_mask=reset)
    reset[:, 3] = True
    decay_cut, _ = mamba_shard_summary(x, b_matrix, delta, a_log, reset_mask=reset)
    assert bool((decay_open > 0).all())
    assert torch.equal(decay_cut, torch.zeros_like(decay_cut))


def test_halo_and_incoming_state_are_inert_without_a_group() -> None:
    disabled = ContextParallelContext.disabled(4)
    values = torch.randn(2, 4, 6)
    assert torch.equal(conv_left_halo(values, disabled, width=3), torch.zeros(2, 2, 6))
    assert torch.equal(
        left_halo(values, disabled, width=2, fill=-1.0), torch.full((2, 2, 6), -1.0)
    )
    state = torch.randn(2, 3, 4, 5)
    assert torch.equal(
        mamba_incoming_state(torch.ones(2, 3), state, disabled), torch.zeros_like(state)
    )


def test_packed_metadata_can_continue_a_neighbouring_shard() -> None:
    """A shard boundary is not a document boundary unless the group says so."""

    from metis_training.model import ActiveTokenLayout

    layout = ActiveTokenLayout(torch.arange(4), batch_size=1, sequence_length=4)
    document_ids = torch.zeros(1, 4, dtype=torch.long)
    _, fresh = _packed_document_metadata(layout, document_ids, continues_previous=False)
    _, continued = _packed_document_metadata(layout, document_ids, continues_previous=True)
    assert bool(fresh[0, 0])
    assert not bool(continued[0, 0])
    assert torch.equal(fresh[0, 1:], continued[0, 1:])


def test_attention_layout_truncates_keys_at_the_local_causal_bound() -> None:
    """Rank 1 may read rank 0's keys, but never rank 2's or its own future."""

    context = ContextParallelContext(group=object(), size=3, rank=1, local_length=2)
    capacity = 2
    # One document spanning all three shards.
    gathered = torch.tensor([0, 0, 0, 0, 0, 0])
    counts = torch.tensor([2, 2, 2])
    layout = build_context_parallel_attention_layout(
        local_segments=torch.tensor([0, 0]),
        gathered_segments=gathered,
        counts=counts,
        context=context,
        capacity=capacity,
    )
    # Positions 0..3 are rank 0's two keys and rank 1's own two; position 4 and
    # 5 belong to rank 2 and are in the future.
    assert layout.key_indices.tolist() == [0, 1, 2, 3]
    assert layout.cu_seqlens_q.tolist() == [0, 2]
    assert layout.cu_seqlens_k.tolist() == [0, 4]


def test_attention_layout_keeps_documents_apart() -> None:
    context = ContextParallelContext(group=object(), size=2, rank=1, local_length=2)
    layout = build_context_parallel_attention_layout(
        local_segments=torch.tensor([7, 7]),
        gathered_segments=torch.tensor([3, 3, 7, 7]),
        counts=torch.tensor([2, 2]),
        context=context,
        capacity=2,
    )
    # Rank 0 holds a different document, so none of its keys are selectable.
    assert layout.key_indices.tolist() == [2, 3]


def test_packed_segment_keys_separate_batch_rows() -> None:
    """Document 0 of row 0 must not compare equal to document 0 of row 1."""

    document_ids = torch.zeros(2, 3, dtype=torch.long)
    selector = torch.arange(6)
    keys = packed_segment_keys(
        document_ids, selector, batch_size=2, sequence_length=3, stride=8
    )
    assert keys.tolist() == [0, 0, 0, 8, 8, 8]


def test_batch_sharding_recounts_supervised_tokens() -> None:
    """Only the final shard has an unsupervised last position."""

    from metis_training.data import TrainingBatch

    length = 8
    input_ids = torch.arange(length, dtype=torch.long).view(1, length)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    batch = TrainingBatch(
        input_ids=input_ids,
        canonical_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones(1, length, dtype=torch.bool),
        document_ids=torch.zeros(1, length, dtype=torch.long),
        reset_mask=torch.zeros(1, length, dtype=torch.bool),
        phase="phase_a",
        global_token_cursor=0,
        next_global_token_cursor=length,
        non_padding_tokens=length,
        supervised_tokens=length - 1,
    )
    shards = [batch.shard_for_context_parallel(rank, 4) for rank in range(4)]
    assert [shard.supervised_tokens for shard in shards] == [2, 2, 2, 1]
    assert sum(shard.supervised_tokens for shard in shards) == batch.supervised_tokens
    assert torch.equal(
        torch.cat([shard.input_ids for shard in shards], dim=1), input_ids
    )
    assert batch.shard_for_context_parallel(0, 1) is batch


def test_batch_sharding_requires_an_even_split() -> None:
    from metis_training.data import TrainingBatch

    ids = torch.zeros(1, 6, dtype=torch.long)
    batch = TrainingBatch(
        input_ids=ids,
        canonical_ids=ids,
        labels=ids,
        attention_mask=torch.ones(1, 6, dtype=torch.bool),
        document_ids=ids,
        reset_mask=torch.zeros(1, 6, dtype=torch.bool),
        phase="phase_a",
        global_token_cursor=0,
        next_global_token_cursor=6,
        non_padding_tokens=6,
        supervised_tokens=5,
    )
    with pytest.raises(ValueError, match="not divisible"):
        batch.shard_for_context_parallel(0, 4)


def test_layer_recompute_is_priced_like_pass_recompute() -> None:
    """Same replay, same arithmetic; the two differ only in peak memory."""

    from metis_training.metrics import estimate_hardware_flops, estimate_train_flops

    base = Metis16Config.tiny_for_tests()
    model_flops = estimate_train_flops(base, tokens=4096)
    for policy, factor in (("none", 1.0), ("pass", 8 / 6), ("layer", 8 / 6)):
        config = replace(base, activation_recompute_policy=policy)
        assert estimate_hardware_flops(config, tokens=4096) == pytest.approx(
            model_flops * factor
        )


# ---------------------------------------------------------------------------
# Multi-rank: real gloo groups
# ---------------------------------------------------------------------------


def _gloo_available() -> bool:
    return dist.is_available() and dist.is_gloo_available()


def _probe_weights(shape, length):
    generator = torch.Generator().manual_seed(4242)
    return torch.rand((shape[0], length, *shape[2:]), generator=generator)


def _reference_run():
    torch.manual_seed(SEED)
    config = replace(Metis16Config.tiny_for_tests(), activation_recompute_policy="pass")
    model = Metis16ForCausalLM(config)
    model.train()
    input_ids, labels, reset_mask, forced_depths = _batch(config)
    output = model(
        input_ids,
        labels=labels,
        reset_mask=reset_mask,
        curriculum=_curriculum(),
        force_depth=forced_depths,
    )
    hidden = output.final_hidden_state
    (hidden.float() * _probe_weights(hidden.shape, LENGTH)).sum().backward()
    return hidden.detach().float()


def _context_parallel_worker(rank: int, world: int, port: int, path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.new_group(ranks=list(range(world)), backend="gloo")

    torch.manual_seed(SEED)
    config = replace(
        Metis16Config.tiny_for_tests(),
        activation_recompute_policy="layer",
        context_parallel_size=world,
    )
    model = Metis16ForCausalLM(
        config, process_groups=MetisProcessGroups(context=group)
    )
    model.train()

    input_ids, labels, reset_mask, forced_depths = _batch(config)
    shard = LENGTH // world
    window = slice(rank * shard, (rank + 1) * shard)
    output = model(
        input_ids[:, window],
        labels=labels[:, window],
        reset_mask=reset_mask[:, window],
        curriculum=_curriculum(),
        force_depth=forced_depths[:, window],
    )
    hidden = output.final_hidden_state
    probe = _probe_weights(hidden.shape, LENGTH)[:, window]
    # Backward must run: the reduce-scatter half of every gather is what a
    # forward-only check would miss, and a rank that skips one hangs the group.
    (hidden.float() * probe).sum().backward()

    shard_hidden = hidden.detach().float().contiguous()
    gathered = [torch.zeros_like(shard_hidden) for _ in range(world)]
    dist.all_gather(gathered, shard_hidden, group=group)
    if rank == 0:
        torch.save(torch.cat(gathered, dim=1), path)
    dist.barrier(group=group)
    dist.destroy_process_group()


@pytest.mark.skipif(not _gloo_available(), reason="gloo is required for CP tests")
@pytest.mark.parametrize("world", [2, 4])
def test_context_parallel_forward_matches_the_unsharded_model(world, tmp_path) -> None:
    """CP=N over shards must reproduce CP=1 over the whole sequence.

    Forward parity is the tight bar.  Gradients are deliberately not compared
    elementwise: the mHC Sinkhorn amplifies a float32 rounding difference by
    roughly a thousand, so a 1e-7 forward difference from any source -- CP or a
    reassociated sum on one rank -- moves ``mix_logits`` by 1e-4.  Backward is
    still exercised, because it is where the collective symmetry is tested.
    """

    reference = _reference_run()
    path = str(tmp_path / "cp_hidden.pt")
    context = mp.get_context("spawn")
    port = 29_600 + world
    processes = [
        context.Process(
            target=_context_parallel_worker, args=(rank, world, port, path)
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=900)
    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]

    observed = torch.load(path, weights_only=False)
    assert observed.shape == reference.shape
    scale = max(reference.abs().max().item(), 1e-8)
    assert (observed - reference).abs().max().item() / scale < 1e-5


def _sparse_sync_worker(rank: int, world: int, port: int, path: str) -> None:
    from metis_training.model import _sync_sparse_gradient_by_row_owner

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.new_group(ranks=list(range(world)), backend="gloo")
    rows_by_rank = (
        torch.tensor([0, 1, 3]),
        torch.tensor([1, 2, 3]),
    )
    values_by_rank = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
    )
    gradient = torch.sparse_coo_tensor(
        rows_by_rank[rank].unsqueeze(0),
        values_by_rank[rank],
        size=(5, 2),
    ).coalesce()
    synchronized = _sync_sparse_gradient_by_row_owner(
        gradient,
        group=group,
    ).coalesce()
    if rank == 0:
        torch.save(
            {
                "dense": synchronized.to_dense(),
                "rows": synchronized.indices()[0],
            },
            path,
        )
    dist.barrier(group=group)
    dist.destroy_process_group()


@pytest.mark.skipif(not _gloo_available(), reason="gloo is required for sparse sync")
def test_row_owner_sparse_sync_matches_global_average(tmp_path) -> None:
    path = str(tmp_path / "sparse-gradient.pt")
    context = mp.get_context("spawn")
    world = 2
    processes = [
        context.Process(
            target=_sparse_sync_worker,
            args=(rank, world, 29_650, path),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)
    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
    observed = torch.load(path, weights_only=False)
    expected = torch.tensor(
        [
            [0.5, 1.0],
            [5.0, 6.0],
            [4.5, 5.0],
            [8.0, 9.0],
            [0.0, 0.0],
        ]
    )
    torch.testing.assert_close(observed["dense"], expected)
    rows = observed["rows"].tolist()
    assert rows == sorted(set(rows))


def _batch_guard_worker(rank: int, world: int, port: int, path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.new_group(ranks=list(range(world)), backend="gloo")
    config = replace(
        Metis16Config.tiny_for_tests(),
        activation_recompute_policy="layer",
        context_parallel_size=world,
    )
    model = Metis16ForCausalLM(config, process_groups=MetisProcessGroups(context=group))
    message = ""
    try:
        model(torch.zeros(2, 4, dtype=torch.long), curriculum=_curriculum())
    except ValueError as error:
        message = str(error)
    if rank == 0:
        torch.save(message, path)
    dist.barrier(group=group)
    dist.destroy_process_group()


@pytest.mark.skipif(not _gloo_available(), reason="gloo is required for CP tests")
def test_context_parallelism_rejects_multi_row_batches(tmp_path) -> None:
    """Packing interleaves batch rows across shards, so guard rather than corrupt."""

    path = str(tmp_path / "guard.pt")
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_batch_guard_worker, args=(rank, 2, 29_620, path))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=600)
    assert all(process.exitcode == 0 for process in processes)
    assert "micro-batch 1" in torch.load(path, weights_only=False)
