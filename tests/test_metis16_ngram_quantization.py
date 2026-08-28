from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from metis_training.model import CurriculumState, Metis16ForCausalLM
from metis_training.checkpointing import CHECKPOINT_LAYOUT, CHECKPOINT_SCHEMA
from metis_training.contracts import canonical_json_sha256, sha256_file
from metis_training.model_config import Metis16Config
from metis_training.ngram_quantization import (
    NGramQuantizationSpec,
    benchmark_table_collection,
    checkpoint_ngram_tensors,
    compare_model_ngram_losses,
    error_metrics,
    fake_quantize_rows,
    quantize_ngram_table,
)


def _weight(rows: int = 17, width: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20_260_828)
    value = torch.randn(rows, width, generator=generator, dtype=torch.float32)
    # Exercise zero blocks and a scale outlier; both are common failure points
    # in block formats and neither is exposed by all-normal random data.
    value[0, :16] = 0
    value[-1, -1] = 31.0
    return value.to(torch.bfloat16)


def test_fp8_clamps_before_cast_instead_of_serializing_nan_codes():
    weight = torch.tensor([[500.0] * 64, [-500.0] * 64], dtype=torch.bfloat16)
    snapshot = quantize_ngram_table(
        weight, NGramQuantizationSpec("fp8_e4m3", block_size=0)
    )
    restored = snapshot.dequantize()
    assert torch.isfinite(restored).all()
    assert not bool(((snapshot.payload == 0x7F) | (snapshot.payload == 0xFF)).any())


@pytest.mark.parametrize("format_name", ["bf16", "fp8_e4m3", "nvfp4"])
def test_every_format_rejects_nonfinite_source_tables(format_name):
    weight = _weight(rows=2)
    weight[1, 1] = torch.inf
    with pytest.raises(ValueError, match="non-finite"):
        quantize_ngram_table(weight, format_name)


def test_nvfp4_representable_values_roundtrip_exactly():
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    row = torch.cat((levels, -levels)).repeat(4).view(1, 64)
    snapshot = quantize_ngram_table(row, "nvfp4")
    torch.testing.assert_close(snapshot.dequantize(), row, rtol=0, atol=0)


def test_storage_reports_include_payload_and_every_scale():
    weight = _weight(rows=2)
    bf16 = quantize_ngram_table(weight, "bf16")
    fp8 = quantize_ngram_table(
        weight, NGramQuantizationSpec("fp8_e4m3", block_size=64)
    )
    nvfp4 = quantize_ngram_table(weight, "nvfp4")

    assert bf16.storage_bytes == 2 * 64 * 2
    assert fp8.storage_bytes == 2 * 64 + 2 * 2
    assert nvfp4.storage_bytes == 2 * 32 + 2 * 4 + 4
    assert bf16.bits_per_parameter == 16.0
    assert fp8.bits_per_parameter == 8.25
    # The one per-table FP32 scale is visible at this tiny size; at production
    # scale this converges to the expected 4.5 bits/parameter.
    assert nvfp4.bits_per_parameter == 4.75


@pytest.mark.parametrize(
    "spec",
    [
        NGramQuantizationSpec("bf16"),
        NGramQuantizationSpec("fp8_e4m3", block_size=0),
        NGramQuantizationSpec("fp8_e4m3", block_size=64),
        NGramQuantizationSpec("nvfp4"),
    ],
)
def test_lookup_dequantizes_only_the_requested_rows(spec):
    weight = _weight()
    snapshot = quantize_ngram_table(weight, spec, chunk_rows=3)
    row_ids = torch.tensor([[16, 0], [5, 5]], dtype=torch.long)
    expected = snapshot.dequantize(output_dtype=torch.bfloat16).index_select(
        0, row_ids.reshape(-1)
    ).view(2, 2, 64)
    observed = snapshot.lookup(row_ids)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    with pytest.raises(IndexError):
        snapshot.lookup(torch.tensor([17]))


def test_stochastic_nvfp4_is_seeded_and_nearest_is_not_silently_reused():
    weight = _weight()
    first = quantize_ngram_table(
        weight, NGramQuantizationSpec("nvfp4", rounding="stochastic", seed=7)
    )
    replay = quantize_ngram_table(
        weight, NGramQuantizationSpec("nvfp4", rounding="stochastic", seed=7)
    )
    other = quantize_ngram_table(
        weight, NGramQuantizationSpec("nvfp4", rounding="stochastic", seed=8)
    )
    nearest = quantize_ngram_table(weight, "nvfp4")
    assert torch.equal(first.payload, replay.payload)
    assert not torch.equal(first.payload, other.payload)
    assert not torch.equal(first.payload, nearest.payload)


def test_fake_quantization_has_a_straight_through_gradient():
    rows = _weight(rows=4).float().requires_grad_(True)
    quantized = fake_quantize_rows(rows, "nvfp4")
    assert not torch.equal(rows.detach(), quantized.detach())
    quantized.sum().backward()
    torch.testing.assert_close(rows.grad, torch.ones_like(rows), rtol=0, atol=0)


def test_error_metrics_detect_a_payload_mutation():
    weight = _weight()
    snapshot = quantize_ngram_table(weight, "nvfp4")
    before = snapshot.dequantize()
    baseline = error_metrics(weight, before)
    snapshot.payload[3, 0] ^= 0x07
    after = snapshot.dequantize()
    mutated = error_metrics(weight, after)
    assert mutated["relative_l2_error"] > baseline["relative_l2_error"]


def test_collection_benchmark_measures_concatenation_projection_and_storage():
    tables = {f"o{order}_h{head}": _weight(rows=19 + head) for order in (2, 3) for head in range(2)}
    generator = torch.Generator().manual_seed(91)
    projection = torch.randn(32, 4 * 64, generator=generator)
    report = benchmark_table_collection(
        tables,
        formats=(
            "bf16",
            NGramQuantizationSpec("fp8_e4m3", block_size=64),
            "nvfp4",
        ),
        projection_weight=projection,
        lookup_rows=64,
        warmup_iterations=0,
        timed_iterations=2,
        chunk_rows=7,
    )
    assert report["table_count"] == 4
    rows = {row["format"]: row for row in report["formats"]}
    assert rows["bf16"]["retrieval_error"]["relative_l2_error"] == 0.0
    assert rows["fp8_e4m3"]["compression_vs_bf16"] > 1.9
    assert rows["nvfp4"]["compression_vs_bf16"] > 3.0
    assert rows["fp8_e4m3"]["projected_error"]["relative_l2_error"] > 0.0
    assert rows["nvfp4"]["projected_error"]["relative_l2_error"] > 0.0
    assert all(row["tokens_per_second"] > 0 for row in report["formats"])
    expected_bf16_bytes = sum(64 * 2 for _ in tables)
    assert rows["bf16"]["retrieved_bytes_per_token"] == expected_bf16_bytes

    malformed = dict(tables)
    malformed["not_a_table"] = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="rank two"):
        benchmark_table_collection(malformed, timed_iterations=1)


def test_checkpoint_loader_selects_only_ngram_tables_and_projection(tmp_path: Path):
    table = _weight(rows=3)
    projection = torch.randn(32, 64)
    path = tmp_path / "state.pt"
    torch.save(
        {
            "model": {
                "layers.0.weight": torch.randn(2, 2),
                "ngram_memory.tables.o2_h0.embedding.weight": table,
                "ngram_memory.projection.weight": projection,
            }
        },
        path,
    )
    tables, loaded_projection = checkpoint_ngram_tensors(path)
    assert list(tables) == ["ngram_memory.tables.o2_h0.embedding.weight"]
    torch.testing.assert_close(tables[next(iter(tables))], table)
    torch.testing.assert_close(loaded_projection, projection)


def test_checkpoint_loader_reads_one_complete_owner_from_chunked_checkpoint(
    tmp_path: Path,
):
    table = _weight(rows=3)
    projection = torch.randn(32, 64)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    items = [
        {
            "kind": "model_tensor",
            "name": "ngram_memory.tables.o2_h0.embedding.weight",
            "component": None,
            "state_key": None,
            "shape": list(table.shape),
            "dtype": str(table.dtype),
            "total_numel": table.numel(),
            "start": 0,
            "end": table.numel(),
            "tensor": table.reshape(-1),
        }
    ]
    table_artifact = state_dir / "tables-ep-0000-shard-00000.pt"
    torch.save(
        {
            "schema": CHECKPOINT_LAYOUT,
            "owner": "tables-ep-0000",
            "shard_index": 0,
            "items": items,
        },
        table_artifact,
    )
    projection_artifact = state_dir / "replicated-shard-00000.pt"
    torch.save(
        {
            "schema": CHECKPOINT_LAYOUT,
            "owner": "replicated",
            "shard_index": 0,
            "items": [
                {
                    "kind": "model_tensor",
                    "name": "ngram_memory.projection.weight",
                    "component": None,
                    "state_key": None,
                    "shape": list(projection.shape),
                    "dtype": str(projection.dtype),
                    "total_numel": projection.numel(),
                    "start": 0,
                    "end": projection.numel(),
                    "tensor": projection.reshape(-1),
                }
            ],
        },
        projection_artifact,
    )

    def inventory_row(owner: str, name: str, value: torch.Tensor) -> dict[str, object]:
        return {
            "owner": owner,
            "kind": "model_tensor",
            "name": name,
            "component": None,
            "state_key": None,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }

    def artifact_row(owner: str, path: Path) -> dict[str, object]:
        return {
            "path": str(path.relative_to(tmp_path)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": "state_shard",
            "owner": owner,
            "item_count": 1,
            "staged_tensor_bytes": path.stat().st_size,
        }

    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "layout": CHECKPOINT_LAYOUT,
        "artifacts": [
            artifact_row("tables-ep-0000", table_artifact),
            artifact_row("replicated", projection_artifact),
        ],
        "state_inventory": [
            inventory_row(
                "tables-ep-0000",
                "ngram_memory.tables.o2_h0.embedding.weight",
                table,
            ),
            inventory_row(
                "replicated", "ngram_memory.projection.weight", projection
            ),
        ],
    }
    manifest["checkpoint_sha256"] = canonical_json_sha256(manifest)
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    tables, loaded_projection = checkpoint_ngram_tensors(tmp_path)
    torch.testing.assert_close(tables[next(iter(tables))], table)
    torch.testing.assert_close(loaded_projection, projection)
    with pytest.raises(KeyError, match="Unknown N-gram checkpoint owner"):
        checkpoint_ngram_tensors(tmp_path, checkpoint_owner="tables-ep-0099")


def _quantizable_tiny_model() -> tuple[Metis16ForCausalLM, Metis16Config]:
    config = Metis16Config.tiny_for_tests()
    config = replace(
        config,
        ngram_memory=replace(config.ngram_memory, value_dim=16),
    )
    config._validate_tiny()
    torch.manual_seed(101)
    # Production N-gram parameters are BF16.  Keeping the fixture in FP32
    # would make the BF16 control itself a quantization treatment.
    model = Metis16ForCausalLM(config)
    for table in model.ngram_memory.tables.values():
        table.embedding.to(dtype=torch.bfloat16)
    model.ngram_memory.projection.to(dtype=torch.bfloat16)
    model.eval()
    return model, config


def _model_batch(config: Metis16Config) -> dict[str, torch.Tensor | CurriculumState]:
    generator = torch.Generator().manual_seed(102)
    input_ids = torch.randint(0, config.vocab_size, (2, 12), generator=generator)
    labels = torch.roll(input_ids, -1, dims=1)
    labels[:, -1] = -100
    return {
        "input_ids": input_ids,
        "labels": labels,
        "canonical_ids": input_ids,
        "curriculum": CurriculumState(
            continuation_mode="fixed_max",
            max_passes=2,
            routed_k_mode="fixed",
            fixed_routed_k=2,
            stochastic_routing=False,
        ),
    }


def test_model_loss_context_changes_only_tables_and_always_restores_bf16():
    model, config = _quantizable_tiny_model()
    batch = _model_batch(config)
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        baseline = model(**batch).loss
        with model.ngram_memory.quantized_table_lookup("bf16") as report:
            exact = model(**batch).loss
            assert report["table_count"] == 4
            assert all(
                table._quantized_snapshot is not None
                for table in model.ngram_memory.tables.values()
            )
        with model.ngram_memory.quantized_table_lookup("nvfp4"):
            compressed = model(**batch).loss
    torch.testing.assert_close(exact, baseline, rtol=0, atol=0)
    assert torch.isfinite(compressed)
    assert all(
        table._quantized_snapshot is None
        for table in model.ngram_memory.tables.values()
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)


def test_model_loss_report_replays_identical_batches_and_training_rejects_snapshot():
    model, config = _quantizable_tiny_model()
    batch = _model_batch(config)
    report = compare_model_ngram_losses(
        model,
        [batch, batch],
        formats=(NGramQuantizationSpec("fp8_e4m3", block_size=16), "nvfp4"),
    )
    assert report["batches"] == 2
    assert report["bf16_losses"][0] == report["bf16_losses"][1]
    assert {row["format"] for row in report["formats"]} == {"fp8_e4m3", "nvfp4"}
    assert all(row["loss_relative_error_vs_bf16"] >= 0 for row in report["formats"])
    assert all(len(row["paired_loss_deltas"]) == 2 for row in report["formats"])
    assert all(row["perplexity_ratio_vs_bf16"] > 0 for row in report["formats"])
    assert all(
        table._quantized_snapshot is None
        for table in model.ngram_memory.tables.values()
    )

    model.train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        model.ngram_memory.enable_quantized_table_lookup("nvfp4")
    assert all(
        table._quantized_snapshot is None
        for table in model.ngram_memory.tables.values()
    )

    model.eval()
    model.ngram_memory.enable_quantized_table_lookup("nvfp4")
    model.train()
    with pytest.raises(RuntimeError, match="cannot be used for training"):
        model(**batch)
    model.ngram_memory.disable_quantized_table_lookup()
