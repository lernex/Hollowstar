# Metis-1.6 N-gram table quantization

This harness answers a deliberately narrow question: how much accuracy and
storage changes when only the sparse N-gram table is changed from BF16 to FP8
E4M3 or NVFP4? It does not quantize the fusion projection, gates, backbone, or
LM head.

## What is measured

The storage report counts the bytes actually retained by the snapshot:

- BF16: 16-bit table values.
- FP8 E4M3: 8-bit values plus BF16 scales. The default uses one scale per
  64-value row; `--fp8-block-size 0` selects one table-wide scale.
- NVFP4: two packed E2M1 values per byte, one E4M3 scale per 16 values, and one
  FP32 global scale per table shard. At production size this approaches 4.5
  bits per table parameter.

The benchmark retrieves one independently hashed row from every table, then
reports error before and after the unchanged learned fusion projection. It also
reports packed bytes, bits per table parameter, compression versus BF16, and
lookup timing. Timing uses the software reference dequantizer. It is valid for
fidelity and storage decisions on Portage, but it is not evidence of native
NVFP4 throughput; that requires a fused kernel on Blackwell-class hardware.

## Checkpoint-table benchmark

Run the complete local table shard, not a synthetic matrix:

```bash
python scripts/metis16_ngram_quant_benchmark.py \
  --checkpoint "$CHECKPOINT/tokens-0000000000000" \
  --device cuda \
  --formats bf16,fp8_e4m3,nvfp4 \
  --fp8-block-size 64 \
  --lookup-rows 65536 \
  --output "$REPORT_ROOT/ngram-quant-table-report.json"
```

Metis-1.6 production checkpoints are row-sharded. The command automatically
loads the first `tables-ep-*` owner and validates the checkpoint manifest and
artifact checksums. Use `--checkpoint-owner tables-ep-XXXX` to repeat the probe
on other owners. Each owner is the table footprint and lookup surface for one
expert-parallel rank; do not describe one owner's `storage_bytes` as the global
table total.

For a cheap smoke test, add `--max-table-rows 65536`. The JSON then sets
`full_tables_benchmarked` to false, so sampled storage cannot be mistaken for
the real footprint. Remove that option for the decision run.

Run a second NVFP4 pass with stochastic rounding rather than silently treating
nearest rounding as the only result:

```bash
python scripts/metis16_ngram_quant_benchmark.py \
  --checkpoint "$CHECKPOINT/tokens-0000000000000" \
  --device cuda \
  --formats nvfp4 \
  --nvfp4-rounding stochastic \
  --lookup-rows 65536 \
  --output "$REPORT_ROOT/ngram-nvfp4-stochastic-report.json"
```

## Model-loss parity

Table-vector error is necessary but not sufficient. Replay the exact same
held-out batches through the trained model and swap only the table lookup:

```python
from metis_training.ngram_quantization import (
    NGramQuantizationSpec,
    compare_model_ngram_losses,
)

model.eval()
report = compare_model_ngram_losses(
    model,
    held_out_batches,
    formats=(
        NGramQuantizationSpec("fp8_e4m3", block_size=64),
        NGramQuantizationSpec("nvfp4", rounding="nearest"),
        NGramQuantizationSpec("nvfp4", rounding="stochastic", seed=16062026),
    ),
)
```

The context manager builds packed snapshots after device placement, changes no
registered parameter or checkpoint value, and restores normal BF16 table lookup
even if evaluation raises. It refuses training mode. The returned report keeps
the paired losses and storage accounting for every format.

The decision should be made on a fixed, representative held-out token set. Keep
both the signed mean NLL delta and the worst batch delta: a small aggregate
number can hide domain-specific damage. Compare FP8 against BF16 first, then
NVFP4 against both. Storage gain does not compensate for a statistically clear
quality regression unless that trade is explicitly accepted.

## Training-noise probe

`fake_quantize_rows(rows, spec)` is a straight-through estimator for a later
QAT ablation. It quantizes only touched rows and preserves their gradients. It
does not claim packed training storage: the BF16 sparse parameter and optimizer
state remain resident and authoritative.
