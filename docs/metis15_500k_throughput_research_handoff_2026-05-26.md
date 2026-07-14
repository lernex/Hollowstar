# Metis-1.5 500k tok/s Throughput Research Handoff

Date: 2026-05-26

Repo: `/Users/giulianno/Documents/10M model`

Primary goal: explain why Metis-1.5 production-shaped training is stuck around the same `30k-40k tok/s` class on very different hardware, and find a credible path to `500k tok/s` sustained production training throughput.

This file is intentionally large and blunt. It is meant for a stronger research model to ingest as much context as possible before proposing fixes. It includes exact model shape, trainer code paths, hardware instances, benchmark results, compile/runtime failures, suspicious coincidences, and a recommended investigation agenda.

## Table Of Contents

1. Executive summary
2. Most important conclusion
3. Throughput scoreboard
4. ETA math for 50B tokens
5. Why the 5M tok/s Trainium number was not the real training number
6. Current canonical Metis-1.5 model shape
7. Hardware and environment matrix
8. Data and S3 layout
9. Current Neuron/Trainium trainer design
10. Trainium experiment timeline and logs
11. H100 experiment history
12. RTX PRO 6000 Blackwell experiment history
13. The cross-hardware 30k-40k phenomenon
14. Likely bottlenecks and hypotheses
15. Code-path details that matter
16. Exact commands and run shapes
17. What has already been tried
18. What not to conclude
19. High-value research questions
20. Candidate routes to 500k tok/s
21. Verification checklist for any proposed fix
22. Appendix: file map

## 1. Executive Summary

The headline problem:

- Synthetic Trainium probes produced multi-million tok/s numbers.
- Real Metis-shaped training did not.
- The best real-ish Trainium cached datapoint so far is only about `37k tok/s`, and that is for a one-layer seq1024 graph, not the full 19-layer model.
- The best stable H100 single-latent BF16 path is about `30.7k-31.1k tok/s`.
- Earlier H100 grouped-GEMM BF16 runs reached about `33.5k-34.0k tok/s`.
- Earlier RTX PRO 6000 Blackwell paths were also in the same broad band:
  - about `28.7k tok/s` in an FP8/BF16-expert packet
  - about `29.7k-37.2k tok/s` after bucketed/reverse-combine/TE SwiGLU changes in short runs
  - about `15.4k tok/s` for a reduced NVFP4/FP8-block stable RTX run on an older/pre-reset path

That is the weird phenomenon: `16-chip Trainium 1`, `1x H100`, and `1x RTX PRO 6000 Blackwell` have all landed in or near the same throughput class once the real Metis MoE training path enters the picture.

This strongly suggests the main blocker is not simply raw memory bandwidth or raw tensor-core peak. It is probably the common training dataflow:

- MoE route/top-k/sort/pack/unpack overhead
- expert dispatch and combine overhead
- tiny and fragmented expert GEMMs
- many launches / graph ops / compiler instructions
- unfused activation and routing kernels
- dynamic shape or pseudo-static capacity pain
- dense replicated components, especially attention and lm head/loss
- low arithmetic intensity at the tested microbatch shapes
- framework/compiler limitations for the exact Metis graph

The current Trainium path did prove expert parallelism in the sense that routed experts are sharded across Neuron ranks and payloads move with all-to-all. It did not prove that this raw Torch/XLA implementation can train Metis-1.5 efficiently.

The target:

```text
500,000 tokens/s sustained production training
50,000,000,000 tokens / 500,000 tokens/s = 100,000 seconds
= 27.78 hours
= 1.16 days of pure training time
```

At the current `30k-40k tok/s` class, 50B tokens is roughly `14.5-19.3 days` before eval/checkpoint/interruption overhead. At `15.4k tok/s`, it is `37.6 days`.

The current state is not close enough for flag tuning. We need a structural throughput fix.

## 2. Most Important Conclusion

The `5M+ tok/s` Trainium result was a synthetic/static probe result. It proved the instance and Neuron compiler can do high-throughput fixed-shape work. It did not include the full Metis training graph:

- no full 19-layer decoder
- no real cross entropy over the full vocab in the same shape
- no full top-k routed MoE path with pack/all-to-all/expert/combine at every layer
- no true production batch/loss/optimizer/checkpoint loop
- no real data memmap loader in the same path
- no same graph complexity as `scripts/train_metis15_neuron.py`

Once we moved to the real Metis trainer, the main blockers became:

- `index_add` dispatch/combine causing Neuron runtime out-of-bounds failures
- compact `gather` dispatch/combine hitting Neuron compiler instruction limits
- `one_hot_gather` hitting compiler transform internal errors
- pure `one_hot` path compiling/running, but producing only about `37k tok/s` cached on a one-layer graph and still dropping/overflowing assignments at capacity factor 4

The research model should treat the synthetic and real numbers as different classes of evidence:

- Synthetic: hardware/compiler can be fast on simple fixed graphs.
- Real Metis: current implementation graph is not fast and has compiler/runtime blockers.

## 3. Throughput Scoreboard

### 3.1 Trainium / Neuron

Current Trainium instance lineage:

- First Trainium box: `54.224.107.24`, spot, terminated.
- Replacement Trainium box: `3.90.245.200`, later SSH timed out during/after a long compile.
- Instance type: `trn1.32xlarge`.
- Hardware observed on first box:
  - 16 Trainium devices
  - 32 NeuronCores
  - 128 vCPUs
  - about 495 GiB RAM
  - four 1.7 TB NVMe disks assembled into 7.0 TB RAID0 at `/mnt/trn1`

Synthetic/static probe results:

| Probe | Shape | Result | Meaning |
| --- | --- | ---: | --- |
| One-core BF16 matmul smoke | `2048x2048` | about `33.5 TFLOP/s` on one logical NeuronCore | Neuron compiler/runtime alive |
| Single-core dense tiny trainer | synthetic | about `307k tok/s per rank` | not Metis production |
| Single-core static local-expert FFN | synthetic | about `237k tok/s per rank`, about `2.24 TFLOP/s per rank` | not full trainer |
| 32-core local-expert FFN | synthetic | about `7.46M tok/s aggregate`, about `70.4 TFLOP/s aggregate` | static expert-local compute only |
| 32-core dense probe | synthetic | about `8.11M tok/s aggregate` | dense toy graph |
| Metis-shaped routed-expert FFN, one core | synthetic, `d_model=1536`, hidden `1024`, top-k-like 4 experts | about `174k tok/s`, about `4.37 TFLOP/s per core` | closer shape, still not trainer |
| Metis-shaped routed-expert FFN, 32 cores | synthetic | per-rank around `238k tok/s`, about `2.2 TFLOP/s per rank` | static FFN probe, not full trainer |

Real Metis trainer results:

| Run | Layers | Seq | Batch | Grad accum | Dispatch pack | Capacity | Result |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Full/real seq1024 original path | 19 | 1024 | 1 | 1 | `index_add` | cf4 | Neuron runtime OOB failure |
| One-layer seq1024 original path | 1 | 1024 | 1 | 1 | `index_add` | cf4 | Neuron runtime OOB failure |
| One-layer seq1024 | 1 | 1024 | 1 | 1 | `one_hot` | cf4 | ran, first compile step `42 tok/s`; cached later step about `37k tok/s` |
| One-layer seq1024 cached rerun | 1 | 1024 | 1 | 1 | `one_hot` | cf4 | step 3 about `37,286 tok/s`, but valid assignments only `3,728-3,840` of `4,096` |
| One-layer seq1024 | 1 | 1024 | 1 | 1 | `gather` | cf4 | compiler graph-size fail, `NCC_EVRF007` |
| One-layer seq1024 | 1 | 1024 | 1 | 1 | `gather`, `--optlevel=1` | cf4 | still graph-size fail, `7.87M` instructions |
| One-layer seq1024 | 1 | 1024 | 1 | 1 | `one_hot_gather` | cf4 | compiler internal transform fail, `NCC_ITCT901` |
| One-layer seq1024 | 1 | 1024 | 1 | 1 | `one_hot` | cf8 | compile ran over 15 minutes; no completed datapoint |

Important caution: the best Trainium trainer number is one layer. It is not full pretraining throughput.

### 3.2 H100

Two H100 lanes matter:

1. Older exact-shape/multi-head-ish and grouped GEMM experiments.
2. Current single-latent reset path.

Key stable H100 results:

| Path | GPU | Precision | Backend | Batch | Result |
| --- | --- | --- | --- | ---: | ---: |
| Old TE grouped fallback | H100 80GB | BF16 | `te_grouped` | 8 | about `2.1k-2.4k tok/s` |
| New PyTorch grouped GEMM | H100 80GB | BF16 | `torch_grouped` | 8 | about `33.5k-34.0k tok/s`, stable 50 steps |
| New grouped GEMM | H100 80GB | BF16 | `torch_grouped` | 10 | about `34.5k tok/s` before `nan` around step 6 |
| Global FP8 / BF16 experts | H100 80GB | FP8-ish | `torch_grouped` | 8 | `40k-47k tok/s` before failure; invalid as training throughput |
| Single-latent first stable floor | H100 80GB | BF16 | `te_grouped` | 4 | about `8.2k tok/s` after warmup |
| Single-latent guarded baseline | H100 80GB | BF16 | `torch_grouped_safe` | 8 | about `30.7k-31.1k tok/s` through 100 steps |

Important H100 dense comparison:

- A reconstructed dense Metis-1.4 FP8 synthetic benchmark on the same H100 image reached about `146k-147k tok/s`.
- User historical reference for dense 500M-ish H100 training was about `170k-175k tok/s`.
- That means the H100 stack itself can produce far higher throughput on dense/static-ish training.
- The Metis-1.5 MoE path is the tax.

### 3.3 RTX PRO 6000 Blackwell

RTX PRO 6000 results are not perfectly apples-to-apples because some were before the single-latent reset. They are still important because they expose the same class of MoE/backend bottleneck.

Key observed RTX results:

| Path | GPU | Precision/backend | Result |
| --- | --- | --- | ---: |
| Early FP8/BF16 expert live best | RTX PRO 6000 Blackwell | FP8 path, BF16 experts | about `28.7k tok/s` |
| Bucketed + reverse combine before TE SwiGLU | RTX PRO 6000 Blackwell | grouped dispatch | about `29.7k tok/s` |
| Old atomic combine | RTX PRO 6000 Blackwell | grouped dispatch | about `29.6k tok/s` |
| Reverse combine + TE SwiGLU | RTX PRO 6000 Blackwell | grouped dispatch | about `37.2k tok/s` short run |
| b18/g11 reverse combine + TE SwiGLU | RTX PRO 6000 Blackwell | grouped dispatch | about `37.0k tok/s` short run |
| Stable reduced NVFP4 / FP8-block handoff path | RTX PRO 6000 Blackwell | reduced NVFP4 plus FP8-block surfaces | about `15.4k tok/s` |

The RTX profile showed extreme fragmentation:

- `cudaLaunchKernel`: `248,586` calls in profile
- `cuLaunchKernelEx`: `110,166` calls
- `cudaLaunchKernelExC`: `52,080` calls
- `cublasLtMatmul`: `137,484` ranges across 36 microbatches
- about `3,819 cublasLt matmul calls per microbatch`
- `moe_grouped_sort` alone consumed more NVTX time than optimizer
- optimizer was not the main time bottleneck in that profile

This profile is one of the strongest clues that the issue is launch/dataflow fragmentation.

## 4. ETA Math For 50B Tokens

The base pretrain target in `configs/metis15_manifest.json` is:

```text
target_train_tokens = 50,000,000,000
```

ETA at selected throughputs:

| Sustained throughput | Pure train seconds | Hours | Days |
| ---: | ---: | ---: | ---: |
| `15,396 tok/s` | `3,247,597` | `902.1` | `37.59` |
| `30,700 tok/s` | `1,628,664` | `452.4` | `18.85` |
| `34,000 tok/s` | `1,470,588` | `408.5` | `17.02` |
| `37,286 tok/s` | `1,340,986` | `372.5` | `15.52` |
| `100,000 tok/s` | `500,000` | `138.9` | `5.79` |
| `250,000 tok/s` | `200,000` | `55.6` | `2.31` |
| `500,000 tok/s` | `100,000` | `27.8` | `1.16` |

Important:

- The `37,286 tok/s` Trainium number is not full 19-layer throughput. It is a one-layer cached datapoint.
- If one naively divided that by 19, full depth would look awful. That linear extrapolation is not reliable because the lm head/loss and compile/runtime overhead are not per-layer in the same way, but it shows why the one-layer result is not sufficient proof.
- No full 19-layer Trainium production-shaped run has completed successfully yet.

## 5. Why The 5M tok/s Trainium Number Was Not The Real Training Number

The 5M+ number came from synthetic/static probes, not production Metis training.

Those probes were useful because they answered:

- Is the instance really Trainium/Neuron?
- Does `torch_xla` compile and run?
- Do all 32 NeuronCores participate?
- Can static dense or expert-local matmul-like workloads reach high aggregate throughput?

They did not answer:

- Can `scripts/train_metis15_neuron.py` compile the full graph?
- Can the real top-k router, all-to-all, expert dispatch, combine, lm head, loss, backward, optimizer, and real data loop run efficiently?
- Can the real seq1024 19-layer Metis graph fit within Neuron compiler instruction limits?
- Can assignment overflow be eliminated without making compile impossible?

This is the key conceptual split:

```text
Synthetic probe: "Neuron can be fast."
Real trainer:    "Our current Metis graph/path is not fast and sometimes does not compile/run."
```

## 6. Current Canonical Metis-1.5 Model Shape

The current `configs/metis15_manifest.json` describes Metis-1.5 as a single-latent MoE decoder.

Core model:

```json
{
  "name": "Metis-1.5",
  "architecture": "metis_single_latent_moe_decoder",
  "model_type": "metis_single_latent_moe",
  "vocab_size": 32768,
  "block_size": 1024,
  "d_model": 1536,
  "n_layer": 19,
  "n_heads": 24,
  "n_kv_heads": 8,
  "head_dim": 64,
  "intermediate_size": 4096,
  "torch_dtype": "bfloat16",
  "attention_backend": "sdpa",
  "training_mode": "static_dense_pretrain",
  "mor_enabled": false,
  "ffn_type": "single_latent_moe"
}
```

MoE shape:

```json
{
  "moe_num_experts": 32,
  "moe_top_k": 4,
  "moe_shared_experts": 1,
  "moe_num_heads": 1,
  "moe_expert_intermediate_size": 1024,
  "moe_router_latent_size": 512,
  "moe_routed_latent_size": 512,
  "moe_activation": "squared_relu",
  "moe_router_score": "sigmoid",
  "moe_aux_loss_coef": 0.0001,
  "moe_balance_strategy": "aux_loss_free_bias",
  "moe_capacity_alignment": 128,
  "moe_backend": "torch_grouped_safe",
  "moe_dispatch_mode": "bucketed",
  "moe_token_dispatcher_type": "alltoall",
  "moe_expert_parallel_size": 8
}
```

Parameter estimates from current manifest:

```text
estimated_params:                    897,428,576
estimated_active_params:             339,586,144
estimated_active_transformer_params: 289,254,496
```

The current compute audit in the H100 single-latent docs printed:

```text
config_estimate_params: 897,428,576
config_estimate_active_params_depth1: 339,586,144
attention_apps_per_layer: 6,291,456
latent_dim: 512
num_experts: 32
top_k: 4
routing_units_per_token: 4
expert_hidden: 1,024
expert_param_apps_per_assignment: 1,048,576
routed_expert_param_apps_per_layer: 4,194,304
shared_expert_param_apps_per_layer: 3,145,728
latent_projection_apps_per_layer: 1,572,864
router_projection_and_match_apps_per_layer: 16,416
rough_total_param_apps_per_token: 339,526,240
estimated_train_flops_per_token: 2,037,157,440
```

At `500k tok/s`, this rough audit implies:

```text
2,037,157,440 train flops/token * 500,000 tok/s
= 1.01857872e15 train flops/s
= about 1.02 PFLOP/s useful training throughput
```

The research model should verify whether this audit is the right basis for Trainium and for current single-latent Metis. Earlier multi-head accounting bugs caused active compute to be understated, so throughput claims should be tied to both tok/s and a validated param-application audit.

## 7. Hardware And Environment Matrix

### 7.1 Trainium / trn1.32xlarge

First Trainium instance:

```text
public IPv4: 54.224.107.24
login: ec2-user
key: /Users/giulianno/.ssh/aws_codex_builder.pem
instance type: trn1.32xlarge
OS: Amazon Linux 2023
Neuron inventory: 16 devices / 32 NeuronCores
scratch: /mnt/trn1 RAID0 from four 1.7 TB NVMe disks
repo: /mnt/trn1/src/metis
venv: /mnt/trn1/venvs/aws_neuron_venv_pytorch
cache: /mnt/trn1/cache/neuron-cc
```

Replacement Trainium instance:

```text
public IPv4: 3.90.245.200
login intended: ec2-user
key: /Users/giulianno/.ssh/aws_codex_builder.pem
status as of this handoff: SSH timed out during probe
```

Neuron stack installed/observed:

```text
torch:          2.8.0 / 2.8.0+cu128 in Python printouts
torch_xla:      2.8.x
torch_neuronx:  2.8.0.2.12.22436+0f1dac25 in later printout
neuronx_distributed: import OK
PJRT_DEVICE=NEURON
NEURON_RT_NUM_CORES=32
```

### 7.2 H100

H100 instances used in prior work:

```text
host: 18.183.61.208
login: ubuntu
key: ~/.ssh/aws_codex_builder.pem
GPU: NVIDIA H100 80GB HBM3
driver: 580.126.16
repo: /opt/dlami/nvme/metis
image: metis15-h100:torch280-te215-torchgrouped
torch: 2.8.0+cu128
CUDA: 12.8
Transformer Engine: 2.15.0+42b8400
```

Single-latent follow-up host:

```text
host: 13.206.201.56
GPU: 1x NVIDIA H100 80GB HBM3
image: lernex/metis-gpu:metis15-h100-single-latent-v1
```

### 7.3 RTX PRO 6000 Blackwell

RTX instances used in prior work:

```text
host examples: 54.144.73.7, 98.94.18.6
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
compute capability: (12, 0)
VRAM: 97,887 MiB
driver: 595.58.03
CUDA from driver: 13.2
repo: /opt/dlami/nvme/metis/10M-model or /opt/dlami/nvme/metis
```

Important images:

```text
metis15-blackwell-fp8-ngc2604:sm120a
metis15-blackwell-ngc2604-te-main:sm120a
nvcr.io/nvidia/pytorch:26.04-py3
```

Important runtime facts:

```text
PyTorch: 2.12.0a0+0291f960b6.nv26.04
CUDA runtime: 13.2
Transformer Engine: 2.16.0.dev0+76c2a9e in TE-main image
NVFP4 recipe exposed: true
MXFP8 recipe exposed: true
Float8BlockScaling exposed: true
```

But exposed did not mean usable for exact Metis training. MXFP8 still failed in exact training GEMMs, and default NVFP4 failed unless reduced safety flags were used.

## 8. Data And S3 Layout

Base data path on GPU instances:

```text
/opt/dlami/nvme/metis/data/metis15_base
```

Base data path on Trainium instance:

```text
/mnt/trn1/src/metis/data/metis15_base
```

Files:

```text
train.bin: about 94 GiB
val.bin:   about 964 MiB
meta.json: about 91 KiB
```

Known metadata:

```text
train_tokens: 50,000,000,000
val_tokens: 505,050,506
vocab_size: 32768
dtype: uint16
```

S3 root used by launchers:

```text
s3://lernex-metis-artifacts-151025633969-us-east-1/metis15
```

Base pretrain data URI convention:

```text
$METIS15_S3_ROOT/pretrain-shards/base
```

Base Neuron checkpoint URI convention:

```text
$METIS15_S3_ROOT/checkpoints/base-neuron
```

Data-loading caution:

- The GPU trainer has a pinned CUDA prefetcher and uses async copies.
- The Neuron trainer uses NumPy memmap windows and sends tensors to the XLA device in `get_batch`.
- The Trainium cached step suggests data is probably not the first-order bottleneck, but the Neuron loader is not a highly tuned streaming input pipeline.
- Any real throughput claim must separate compile time, first-step warmup, data wait, forward/backward, optimizer, and checkpoint/eval.

## 9. Current Neuron/Trainium Trainer Design

Main file:

```text
scripts/train_metis15_neuron.py
```

Launcher:

```text
scripts/metis15_neuron_pretrain.sh
make metis15-neuron-pretrain
```

This is a custom raw Torch/XLA trainer, not a full NeuronX Distributed or Megatron-Core port.

Important design choices:

- Uses `torch_xla` PJRT with `PJRT_DEVICE=NEURON`.
- Uses `torchrun --nproc_per_node=32` for one process per NeuronCore.
- Uses `dist.init_process_group("xla")` for distributed init.
- Uses `xm.all_to_all(...)` for static expert-parallel token exchange.
- Uses manual all-reduce of replicated gradients.
- Sharded local routed experts are not all-reduced against unrelated expert shards.
- Uses BF16 on XLA.
- Overrides the manifest to:
  - `attention_backend = "eager"`
  - `native_gqa_attention = False`
  - `low_precision_mode = "none"`
  - `torch_dtype = "bfloat16"`
  - `ffn_type = "single_latent_moe"`
  - `moe_backend = "torch_bmm"`
  - `moe_dispatch_mode = "bucketed"`
  - `moe_memory_efficient_permutation = True`
  - `moe_permute_fusion = False`
  - `moe_fused_combine = True`
  - `moe_balance_bias_update_rate = 0.0`

That last point matters: the Neuron trainer currently disables the aux-loss-free balance-bias update from the manifest. Router balance is therefore not getting the intended adaptive bias update during the run.

### 9.1 Runtime setup

The relevant runtime setup is:

```python
def setup_runtime(device_arg: str) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    wants_xla = device_arg == "xla" or os.environ.get("PJRT_DEVICE", "").upper() == "NEURON"
    if wants_xla:
        import torch_xla.core.xla_model as xm

        if distributed:
            import torch_xla.distributed.xla_backend

            if not dist.is_initialized():
                dist.init_process_group("xla")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        device = xm.xla_device()
        ...
```

All-to-all:

```python
def xla_all_to_all(runtime: Runtime, value: torch.Tensor) -> torch.Tensor:
    if not runtime.is_xla or runtime.world_size == 1:
        return value
    return runtime.xm.all_to_all(value, 0, 0, runtime.world_size, pin_layout=False)
```

Gradient handling:

```python
def xla_all_reduce_gradients(runtime: Runtime, model: nn.Module, *, sharded_name_fragment: str = ".moe.experts.") -> None:
    if runtime.world_size <= 1:
        return
    replicated_grads: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if sharded_name_fragment in name:
            param.grad.mul_(1.0 / float(runtime.world_size))
        else:
            replicated_grads.append(param.grad)
    if replicated_grads:
        runtime.xm.all_reduce(runtime.xm.REDUCE_SUM, replicated_grads, scale=1.0 / float(runtime.world_size), pin_layout=False)
```

### 9.2 Static Expert Parallel MoE

Class:

```text
StaticExpertParallelMoE
```

Key facts:

- Global routed experts: `32`.
- `world_size=32` on trn1.32xlarge.
- Local routed experts per rank: `32 / 32 = 1` in the Trainium run.
- Top-k assignments per token: `4`.
- Shared expert is replicated.
- Router and dense components are replicated.

The core dispatch flow:

1. Project hidden states from `d_model=1536` to routed latent dim `512`.
2. Route tokens with sigmoid scores and top-k over 32 experts.
3. Flatten assignments: `num_rows * top_k`.
4. Compute owner rank from expert id.
5. Pack fixed-capacity send buffers for every destination rank.
6. `xm.all_to_all` hidden payloads to owner ranks.
7. Run local expert(s).
8. `xm.all_to_all` outputs back.
9. Combine outputs back to source token rows.

### 9.3 Capacity formula

Code:

```python
def _capacity(self, assignments: int) -> int:
    if self.explicit_capacity is not None and self.explicit_capacity > 0:
        capacity = int(self.explicit_capacity)
    else:
        per_rank = math.ceil(float(assignments) / float(max(1, self.runtime.world_size)))
        capacity = int(math.ceil(per_rank * max(self.capacity_factor, 1.0)))
    if capacity % self.capacity_alignment:
        capacity += self.capacity_alignment - (capacity % self.capacity_alignment)
    return max(1, min(capacity, assignments))
```

For the one-layer Trainium tests:

```text
batch_size = 1
seq_len = 1024
top_k = 4
assignments per source rank = 1 * 1024 * 4 = 4096
world_size = 32
balanced assignments per destination = 4096 / 32 = 128
capacity factor 4.0 -> 512 slots per destination after alignment
capacity factor 8.0 -> 1024 slots per destination after alignment
```

At cf4, valid assignments still logged only `3,728-3,840` of `4,096`. That means initial router imbalance overflowed at least one destination even with 4x the balanced capacity.

This is suspicious because it means:

- Random router initialization may be very imbalanced.
- Balance bias update is disabled in the Neuron trainer.
- Capacity overflow/drop is active in the only fast-ish Trainium datapoint.
- Raising capacity makes compile much worse.

### 9.4 Dispatch pack implementations

The current parser exposes:

```python
parser.add_argument(
    "--dispatch-pack-impl",
    choices=["index_add", "one_hot", "gather", "one_hot_gather"],
    default="index_add",
)
```

Implementation meanings:

- `index_add`: original pack/combine path using `index_add`.
- `one_hot`: replaces pack and combine with selector matrices and matmul-style gather/combine to avoid XLA scatter OOB.
- `gather`: compact path using argsort/gather/searchsorted to avoid huge one-hot selectors.
- `one_hot_gather`: one-hot packer plus compact gather/searchsorted combine.

Observed behavior:

| Impl | CPU equivalence | Neuron one-layer seq1024 | Comment |
| --- | --- | --- | --- |
| `index_add` | baseline | runtime OOB | unsafe on Neuron |
| `one_hot` | passed | compiles/runs | slow and overflows assignments at cf4 |
| `gather` | passed | compiler instruction limit | better algorithmically but compiler too large |
| `one_hot_gather` | passed | compiler internal error | hybrid failed in Neuron compile |

### 9.5 Why `one_hot` is probably intrinsically bad

For each source rank at seq1024/batch1:

```text
assignments = 4096
capacity = 512 at cf4
selector per destination = [512, 4096] = 2,097,152 elements
destinations = 32
selector elements per layer per rank = 67,108,864
```

That selector is then used in matmuls to pack:

- hidden payload
- weights
- local expert ids
- source rows
- assignment ids
- valid flags

This avoids the Neuron scatter OOB but pays a gigantic graph/dataflow cost. It is plausible that this explains why Trainium collapses into the same `30k-40k tok/s` range despite lots of chips.

### 9.6 Logging throughput calculation

In `train_metis15_neuron.py`:

```python
tokens_per_step = args.batch_size * args.grad_accum_steps * config.block_size * runtime.world_size
...
interval_tokens += args.batch_size * args.grad_accum_steps * config.block_size
tokens_seen += tokens_per_step
...
total_tokens = interval_tokens * runtime.world_size
global_tokens_per_s = total_tokens / elapsed
```

So logged `tok/s` is global tokens per second across all ranks. For the Trainium one-layer run:

```text
tokens per logged step = 1 * 1 * 1024 * 32 = 32,768
step_s 0.879 -> 37,286 tok/s
```

Compile and XLA graph materialization can dominate early steps. Cached steps are the only meaningful short throughput datapoints.

## 10. Trainium Experiment Timeline And Logs

### 10.1 Instance bring-up

Key steps:

1. User provided IPv4 `54.224.107.24`.
2. `ubuntu@...` failed.
3. `ec2-user@...` with `/Users/giulianno/.ssh/aws_codex_builder.pem` succeeded.
4. Instance inventory confirmed `trn1.32xlarge`.
5. Four NVMe disks were assembled into `/mnt/trn1`.
6. Repo was copied to `/mnt/trn1/src/metis`.
7. Neuron venv was created at `/mnt/trn1/venvs/aws_neuron_venv_pytorch`.
8. Neuron cache used `/mnt/trn1/cache/neuron-cc`.
9. Real data was hydrated at `/mnt/trn1/src/metis/data/metis15_base`.

### 10.2 Neuron synthetic probe

File added:

```text
scripts/neuron_synthetic_bench.py
```

It has two main probes:

- `DenseBench`: embedding, one MLP, lm head, CE, AdamW.
- `LocalExpertBench`: local expert FFN-like matmuls over static slices.

The synthetic benchmark does not include the full Metis routing/all-to-all/loss graph.

### 10.3 Full 19-layer seq1024 index_add attempt

Shape:

```text
world_size = 32
layers = 19
block_size = 1024
batch_size = 1
grad_accum = 1
dispatch_pack_impl = index_add
expert_capacity_factor = 4.0
real data = yes
```

Observed failure:

```text
NRT_EXEC_OOB
failed to run scatter/gather (indirect memory copy via vector DGE), due to out-of-bound access
model.MODULE_9560609358915950792+14bf702a.neff
```

Interpretation:

- The original scatter/combine path was not safe on Neuron.
- This was not just a full-depth problem, because a one-layer `index_add` run also failed.

### 10.4 One-layer seq1024 index_add attempt

Observed failure:

```text
failed to run embedding table update, due to out-of-bound access
```

Interpretation:

- The OOB failure follows the dispatch/combine shape even when depth is reduced.
- Need to avoid `index_add`/scatter-like pattern on Neuron or prove a safe bounded version.

### 10.5 One-layer seq1024 one_hot cf4 success

Shape:

```text
world_size = 32
layers = 1
block_size = 1024
batch_size = 1
grad_accum = 1
dispatch_pack_impl = one_hot
expert_capacity_factor = 4.0
real data = yes
constant lr = yes in smoke
skip checkpoint = yes in smoke
```

First compile-heavy run:

```text
step 1 | loss 10.7198 | valid_assign 3,808 | tok/s 42 | step_s 773.258
```

Cached rerun later:

```text
step 1 | tok/s 3,609 | step_s 9.079
step 2 | tok/s 39    | step_s 837.089
step 3 | tok/s 37,463 | step_s 0.875
```

Current rerun after reboot/code changes:

```text
step      1 | loss 10.7198 | lm 10.7198 | moe_aux 0.00041 | valid_assign 3,808 | tok/s 3,715 | step_s 8.820 | lr 1.000000e-05 | tok_seen 0.0000B
step      2 | loss 9.9358  | lm 9.9358  | moe_aux 0.00041 | valid_assign 3,840 | tok/s 4,320 | step_s 7.585 | lr 1.000000e-05 | tok_seen 0.0001B
step      3 | loss 9.3891  | lm 9.3891  | moe_aux 0.00051 | valid_assign 3,728 | tok/s 37,286 | step_s 0.879 | lr 1.000000e-05 | tok_seen 0.0001B
```

Interpretation:

- This is the best real Metis/Trainium trainer datapoint so far.
- It is one layer, not full model.
- It is using `one_hot`, which is likely expensive.
- It is dropping/overflowing assignments at cf4.
- The step-to-step compile/cache behavior is dramatic.

### 10.6 Gather path

Rationale:

- Avoid huge one-hot selector matrices.
- Use argsort/gather/searchsorted to compact payloads and combine outputs.

Initial implementation with `index_select` crashed XLA:

```text
F shape.cc:166 Check failed: state Expected an array shape. Got (f32[512], u32[512])
torch_xla::tensor_methods::index_select
```

Tiny isolated tests found:

- `argsort -> index_select` crashed with tuple-shape errors.
- `argsort -> torch.gather` compiled OK.
- `searchsorted + gather` compiled OK in tiny one-core tests.

Patch:

- Changed gather path to use `torch.gather` instead of `index_select`.
- CPU equivalence passed.
- One-core mini MoE direct `_dispatch_static_ep` forward/backward compiled OK:

```text
xla mini dispatch ok 0.04163239896297455 34.0
```

Full 32-core one-layer seq1024 gather compile then failed:

```text
NCC_EVRF007
Instructions generated by compiler 9,524,436 exceeds typical limit of 5,000,000
```

With `NEURON_CC_FLAGS` including `--optlevel=1`, it still failed:

```text
7,865,414 exceeds typical limit of 5,000,000
```

Interpretation:

- Gather is algorithmically better than one-hot but too much for the Neuron compiler in this raw graph form.
- Need a lower-level/custom/static Neuron-friendly dispatcher, an official NxD/Megatron dispatcher, or a decomposed graph.

### 10.7 one_hot_gather hybrid

Rationale:

- Keep one-hot packer that Neuron already compiled.
- Replace one-hot combine with compact gather/searchsorted combine.

CPU equivalence:

```text
dispatcher equivalence ok
```

Full 32-core one-layer seq1024 compile failed:

```text
NCC_ITCT901 TCTransform assertion error
```

Failure location:

```text
torch.searchsorted(sorted_keys, query, right=False)
```

Interpretation:

- The compiler transform path cannot handle this full graph.
- It is not a runtime throughput candidate yet.

### 10.8 one_hot cf8 attempt

Rationale:

- cf4 still dropped assignments.
- cf8 should allow up to `1024` assignments per destination from each source rank.

Shape:

```text
dispatch_pack_impl = one_hot
expert_capacity_factor = 8.0
layers = 1
block_size = 1024
batch_size = 1
grad_accum = 1
```

Status:

- Compile ran for over 15 minutes.
- No completed throughput datapoint.
- The user interrupted while it was still compiling.
- Later SSH to the replacement host timed out, so no fresh log could be collected.

Interpretation:

- Increasing capacity to avoid overflow may materially increase compile cost and graph size.
- The no-overflow path is not currently operationally proven.

### 10.9 Current Trainium SSH status

At the time this handoff was written, this command was attempted:

```bash
ssh -i /Users/giulianno/.ssh/aws_codex_builder.pem \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  -o ServerAliveInterval=5 \
  -o ServerAliveCountMax=2 \
  ec2-user@3.90.245.200 'hostname'
```

It failed with:

```text
ssh: connect to host 3.90.245.200 port 22: Operation timed out
```

So the latest cf8 outcome is unknown.

## 11. H100 Experiment History

### 11.1 H100 grouped-GEMM backend findings

The first meaningful H100 throughput pass replaced TE `GroupedLinear` with PyTorch `_grouped_mm`.

Important observed schema:

```text
aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
aten::_scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
```

Important details:

- Offsets must be device-side int32 cumulative end offsets.
- Weight layout that works is `[G, K, N]`.
- Empty groups can trigger hangs/bugs in backward.

Results:

```text
TE grouped BF16, batch 8:        ~2.1k-2.4k tok/s
torch_grouped BF16, batch 8:     ~33.5k-34.0k tok/s, stable 50 steps
torch_grouped BF16, batch 10:    ~34.5k tok/s before nan around step 6
global FP8 / BF16 experts:       ~40k-47k tok/s before nan, invalid
torch.compile:                   failed/unusable
```

Key stable H100 batch 8 real-data lines:

```text
step 2:  tok/s 33,150
step 10: tok/s 33,533
step 20: tok/s 33,668
step 30: tok/s 33,826
step 40: tok/s 34,007
step 50: tok/s 33,883
```

Why it still was not enough:

- `~34k tok/s` is far below the then-target `250k tok/s`.
- Expert backend got much better, but still not a fused MoE kernel.
- FP8 was numerically unstable.
- Batch 10 was unstable.
- Batch 12 was OOM or unstable.
- `torch.compile` and CUDA graphs were blocked by dynamic MoE routing, mutable counters, Liger graph breaks, and rotary cache issues.

### 11.2 H100 single-latent reset

Current single-latent H100 handoff:

```text
model_type: metis_single_latent_moe
ffn_type: single_latent_moe
d_model: 1536
layers: 19
experts: 32
top_k: 4
shared experts: 1
router latent size: 512
routed latent size: 512
expert hidden: 1024
attention: SDPA/native GQA
precision: BF16
```

Initial stable floor:

```text
te_grouped, batch 4, 20 steps:
step 20 train 10.3759 tok/s 8,187
```

Guarded native grouped path:

```text
backend: torch_grouped_safe
loss: standard CE
batch: 8
context: 1024
steps: 100
throughput after warmup: ~30.7k-31.1k tok/s
```

Logged:

```text
step  20 tok/s 30,428
step  40 tok/s 30,834
step  60 tok/s 31,057
step  80 tok/s 31,014
step 100 tok/s 30,737
```

Batch matrix:

```text
batch 4, grad_accum 1, 50 steps: stable, ~20.4k-21.0k tok/s after warmup
batch 4, grad_accum 2, 50 steps: stable, ~20.8k-21.4k tok/s after warmup
batch 6, grad_accum 1, 50 steps: stable, ~27.3k-27.5k tok/s after warmup
batch 8, grad_accum 1, 100 steps: stable, ~30.7k-31.1k tok/s after warmup
```

The stable H100 baseline required:

```text
no-empty expert padding
dummy row masking
grouped-output sanitize/clamp
squared-ReLU sanitize/clamp
sync after safe grouped expert GEMMs
standard CE
retained standard-CE logits
```

This should be treated as a guarded baseline, not a clean high-performance backend.

### 11.3 H100 fused SwiGLU attempt

Custom Triton SwiGLU:

- Isolated expert MLP fwd/bwd/AdamW step improved about `1.44x`.
- Full training with fast Triton backward reached `~39k-41k tok/s` before failure.
- Failure persisted even with `lr=0`, suggesting state corruption or unsafe backward, not ordinary optimizer overshoot.

Observed debug:

```text
nonfinite_swiglu surface=grouped_experts gate_up_finite=True gate_up_min=-3.593069e+36 gate_up_max=2.866148e+36 hidden_min=-3.402823e+38 hidden_max=3.402823e+38
```

Conclusion:

- Fused SwiGLU is a real lever.
- The hand-written Triton backward from that pass is not safe enough.
- A verified CUDA/CUTLASS/Triton kernel may still be important.

### 11.4 Dense Metis-1.4 sanity on H100

Dense reconstructed Metis-1.4 FP8 on the same H100 image:

```text
batch 32: mean_tok_s=145,795
batch 40: mean_tok_s=147,446
```

This proves:

- The H100 and software stack can exceed `100k tok/s`.
- Metis-1.5's sparse MoE backend/dataflow is the main tax.
- The old user reference of `~170k-175k tok/s` is plausible in the dense lane.

## 12. RTX PRO 6000 Blackwell Experiment History

### 12.1 Early FP8 throughput research packet

Main finding:

```text
Current best verified RTX PRO 6000 Blackwell FP8 path: ~28.7k tok/s
Target at the time: 170k-200k tok/s
Required speedup to 170k: ~5.9x
Required speedup to 200k: ~7.0x
```

Current best live log in that packet:

```text
train-fp8-b16g24-bf16experts-adamwloop-src-te216main.log
Best max throughput: 28,668 tok/s
Best last throughput: 28,511 tok/s
```

Profile conclusion:

- Not data loading.
- Not primarily optimizer time.
- Main issues were launch/API overhead, GEMM fragmentation, grouped GEMM overhead, sort/gather/unpermute/scatter, unfused SwiGLU, and inefficient all-FP8 expert path.

### 12.2 RTX NSYS profile clue

The RTX profile is crucial because it makes the "same throughput across hardware" phenomenon less mysterious.

Profile around `27.9k tok/s`:

```text
cudaLaunchKernel:              248,586 calls
cuLaunchKernelEx:              110,166 calls
cudaLaunchKernelExC:            52,080 calls
cublasLtMatmul:                137,484 ranges
cublasLtMatmulAlgoGetHeuristic: 137,484 ranges
profile microbatches:               36
cublasLt calls per microbatch:     ~3,819
```

NVTX highlights:

```text
:backward                    21.8%   36 instances
:forward                     18.8%   36 instances
:nvte_multi_tensor_gemm      10.6%   4,104 instances
:nvte_cublas_gemm_v2         10.1%   137,484 instances
:moe_routed_grouped           8.8%   684 instances
:moe_grouped_sort             5.4%   684 instances
:fused_linear_ce              4.6%   36 instances
:optimizer_step               1.5%   3 instances
:batch_fetch                 ~0.0%
```

Interpretation:

- MoE sort and launch fragmentation are very real.
- Optimizer was visible but not first-order time.
- Data loader was not the bottleneck in that profile.
- We were paying thousands of GEMM/API events per microbatch.

### 12.3 Blackwell low-precision failures

RTX PRO 6000 exact-shape kernel smoke found:

- Default NVFP4 failed exact Metis shapes.
- Reduced NVFP4 worked only with:

```text
--nvfp4-disable-rht
--nvfp4-disable-2d-quantization
--nvfp4-disable-stochastic-rounding
```

- MXFP8 was exposed after guard patching but failed in exact training GEMMs.
- BF16 grouped MoE smoke was sometimes faster than reduced NVFP4/FP8-block grouped MoE.

Important warning:

```text
Unfused NVFP4 quantization fallback because input inner dim not multiple of 128, disabling NVFP4 grouped kernel fusion.
```

This warning was for earlier latent dim `320` path, before the current single-latent `512` path. It still matters as a general warning: exact expert dimensions and kernel alignment can dominate low-precision wins.

### 12.4 RTX stable reduced NVFP4 path

Stable RTX handoff setting:

```text
batch: 18
grad_accum: 11
seq_len: 1024
tokens per optimizer step: 202,752
precision: reduced NVFP4 plus FP8-block surfaces
attention: SDPA native GQA
lm loss: Liger fused linear CE
optimizer: Muon-AdamW hybrid with foreach AdamW path
prefetch depth: 4
MoE dispatch: grouped dispatch, capacity factor 0
```

Best stable observed:

```text
step 2 train 10.6460
tok/s 15,396
step_s 13.17
est_tflops 87.85
active_tflops 27.46
```

The stable path was much slower than the short FP8 packet best, but it was a more conservative low-precision training path.

### 12.5 RTX throughput patch

Patch notes showed:

| Run | Final logged tok/s |
| --- | ---: |
| b16 g12, bucketed + reverse combine before TE SwiGLU | `29.7k` |
| b16 g12, old atomic combine | `29.6k` |
| b16 g12, reverse combine + TE SwiGLU | `37.2k` |
| b18 g11, reverse combine + TE SwiGLU | `37.0k` |

Conclusion in that doc:

```text
The patch improves the current family, but it does not make 180k tok/s plausible with TE GroupedLinear still in the hot path.
```

This maps directly to the current Trainium finding: changing memory bandwidth/hardware family did not remove the MoE dispatch/backend wall.

## 13. The Cross-Hardware 30k-40k Phenomenon

Observed:

- RTX PRO 6000 after some dispatch fixes: about `28k-37k tok/s`.
- H100 stable grouped path: about `30.7k-34k tok/s`.
- Trainium one-layer cached real trainer: about `37k tok/s`.

Why this is interesting:

- These systems have very different memory subsystems and tensor hardware.
- Trainium uses 32 NeuronCores across 16 Trainium chips.
- H100 is one 80GB HBM GPU.
- RTX PRO 6000 is one 96GB GDDR7 Blackwell GPU.

If memory bandwidth alone were the primary bottleneck, the curves should look more different. Instead, they are bunched together once the real Metis path enters:

```text
route -> topk -> pack/sort/gather -> expert compute -> combine -> dense/loss/optimizer
```

Most likely interpretation:

1. The workload is not keeping the hardware in large efficient GEMMs.
2. Expert work is fragmented into many small/ragged operations.
3. Dispatch/combine/sort/selector work is dominating enough to erase raw compute differences.
4. Dense replicated parts still matter.
5. Low-precision paths are not fully realized or stable.
6. The tested microbatches are too small to saturate the hardware.
7. Framework/compiler overhead is a first-order term.

Trainium did add expert parallelism, but the implementation still routes through a very expensive fixed-capacity all-to-all graph. Expert parallelism reduced local expert parameter ownership; it did not automatically fuse or accelerate routing/dispatch/combine.

## 14. Likely Bottlenecks And Hypotheses

### Hypothesis A: Current MoE dispatch dominates

Evidence:

- RTX profile showed `moe_grouped_sort` alone above optimizer time.
- RTX profile showed thousands of cublasLt matmul calls per microbatch.
- H100 improved drastically when TE GroupedLinear was replaced, but still stalled around `34k`.
- Trainium one-hot packer builds huge selector graphs.
- Gather alternatives hit compiler instruction limits.

Research direction:

- Replace current route/pack/combine with a real fused/segmented MoE dispatcher.
- Avoid `argsort` if possible; use counting sort/histogram/prefix-sum specialized for 32 experts.
- Use persistent buffers and static capacity only with safe overflow semantics.
- On Trainium, prefer an official NeuronX Distributed/Megatron token dispatcher if available.

### Hypothesis B: Microbatch is too small to saturate hardware

Trainium real trainer currently used:

```text
local batch = 1
seq_len = 1024
world_size = 32
global tokens per step = 32,768
```

At `37,286 tok/s`, step time is about `0.879s`. To reach `500k tok/s` at the same global tokens per step:

```text
32,768 / 500,000 = 0.0655 seconds/step
```

That is unrealistic for the current graph.

If local batch were 8:

```text
global tokens per step = 8 * 1024 * 32 = 262,144
500k tok/s step time target = 0.524 seconds/step
```

Larger batches may be necessary, but:

- GPU paths hit memory/stability limits.
- Trainium compile graph/capacity cost may explode.
- Need measure max batch after dispatcher is fixed.

### Hypothesis C: Dense replicated components are still expensive

Expert parallelism shards routed experts, but these remain replicated:

- embeddings
- attention
- router
- latent down/up projections
- shared expert
- final norm
- lm head / CE

The Neuron trainer does not tensor-parallel shard the dense layers or lm head. If routed expert math is sharded but dense/loss remains replicated and inefficient, EP alone cannot produce 500k tok/s.

Research direction:

- Tensor-parallel dense projections and lm head/loss.
- Sharded vocab/lm head cross entropy.
- Sequence parallel where useful.
- Distributed optimizer for replicated state.

### Hypothesis D: Low-precision is not actually doing useful low-precision work

RTX evidence:

- Reduced NVFP4 sometimes slower than BF16 for expert shapes.
- MXFP8 exposed but failing.
- FP8 H100 unstable in full training.
- Stable H100 lane is BF16.
- Trainium lane is BF16.

Research direction:

- For GPUs, find a stable FP8/FP4 path or prove BF16 + better fused MoE is the shortest route.
- For Trainium, BF16 may be the practical lane unless Neuron supports lower precision for these exact shapes.

### Hypothesis E: Router imbalance and capacity overflow matter

Trainium cf4 one-hot valid assignments:

```text
3,728-3,840 valid out of 4,096 possible assignments
```

This is not acceptable for real training if it means dropping routed expert assignments.

Potential causes:

- Router random init creates early expert imbalance.
- Balance bias update disabled in Neuron trainer.
- Capacity is per destination rank, and one or more destinations overflow.
- `sigmoid` top-k routing may be less balanced at init than expected.

Research direction:

- Implement correct aux-loss-free bias update or other balancing in the Neuron trainer.
- Warm-start balance bias?
- Use top-k routing initialization that spreads experts.
- For throughput tests, temporarily force balanced synthetic expert assignment to isolate dispatch speed from router imbalance.
- Measure per-destination assignment histograms before packing.

### Hypothesis F: Neuron compiler is fighting a graph that should not exist at that level

Evidence:

- `gather` path hit `9.52M` compiler instructions.
- `--optlevel=1` still `7.87M`.
- `one_hot_gather` hit `NCC_ITCT901`.
- cf8 one-hot compile ran for over 15 minutes for one layer.

Research direction:

- Do not keep growing raw Torch/XLA graphs for MoE dispatch.
- Move token dispatch into a smaller custom op / supported Neuron primitive / NxD transformer layer.
- Split compile boundaries if possible.
- Use official NeuronX Distributed MoE patterns.
- Reduce dynamic sort/searchsorted operations from the graph.

### Hypothesis G: The lm head/loss may be a hidden cap

Current model:

```text
vocab_size = 32768
d_model = 1536
tied embeddings = true
lm_head = Linear(1536, 32768)
```

The Neuron trainer computes:

```python
logits = self.lm_head(hidden_states)
shift_logits = logits[:, :-1, :].contiguous().float()
lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
```

That is dense and replicated. On Trainium, it is not vocab-sharded.

Research direction:

- Profile lm head/loss share.
- Test sharded vocab CE.
- Test sampled/parallel CE only if training objective remains correct.
- Compare throughput with dummy loss to bound lm head cost.

## 15. Code-Path Details That Matter

### 15.1 Current manifest forces BF16 EP lane

`configs/metis15_manifest.json` currently says:

```json
"low_precision_mode": "none",
"fp8_expert_precision": "bf16",
"nvfp4_keep_embeddings_bf16": true,
"nvfp4_keep_qkv_bf16": true,
"nvfp4_keep_latent_moe_projections_bf16": true,
"nvfp4_keep_lm_head_bf16": true,
"nvfp4_qkv_precision": "bf16",
"nvfp4_latent_moe_projection_precision": "bf16",
"nvfp4_lm_head_precision": "bf16"
```

This is intentionally conservative after FP8/NVFP4 instability.

### 15.2 Main CUDA trainer MoE path

Relevant file:

```text
src/metis_mamba/model.py
```

Important current surfaces:

- `MetisSingleLatentMoE`
- `torch_grouped_safe`
- `torch_bmm` fallback
- `torch_looped` diagnostic fallback
- bucketed dispatch
- memory-efficient permutation
- no-empty expert padding
- retained standard-CE logits in H100 safe path

The H100 safe path is heavily guarded. It is not proof that `_grouped_mm` is ideal.

### 15.3 Current Neuron trainer is separate and simplified

The Neuron trainer does not directly use the full CUDA model implementation. It reimplements:

- RMSNorm
- eager attention
- squared-ReLU experts
- static EP MoE
- tied lm head
- CE loss
- AdamW

This simplifies graph bring-up, but it means:

- no CUDA Triton bucket dispatcher
- no Liger CE
- no production CUDA prefetcher
- no full optimizer policy
- no Transformer Engine low precision
- no Megatron/NxD fused MoE stack

The research model should not assume the Neuron trainer is a tuned final training stack.

### 15.4 Neuron attention path is eager

The Neuron trainer attention uses explicit matmul/softmax:

```python
scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
causal = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=hidden_states.device, dtype=torch.float32), diagonal=1)
scores = scores.float() + causal.view(1, 1, seq_len, seq_len)
weights = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
out = torch.matmul(weights, value)
```

This is not flash attention. It may become a significant bottleneck at full 19 layers.

### 15.5 Optimizer

GPU production path:

- `muon_adamw`
- custom `MuonAdamWHybrid`
- routed experts usually stay in AdamW
- foreach AdamW path
- optimizer memory can trigger OOM after step 1

Neuron trainer:

- plain `torch.optim.AdamW`
- no Muon
- no sharded optimizer beyond sharded expert parameters
- replicated parameters still have replicated optimizer state per rank

Optimizer is not the primary RTX time bottleneck in the profile, but it is a major memory/scale issue.

## 16. Exact Commands And Run Shapes

### 16.1 Trainium environment

Canonical environment:

```bash
export PATH=/opt/aws/neuron/bin:$PATH
source /mnt/trn1/venvs/aws_neuron_venv_pytorch/bin/activate
cd /mnt/trn1/src/metis

export PJRT_DEVICE=NEURON
export NEURON_RT_NUM_CORES=32
export NEURON_CC_FLAGS="--cache_dir=/mnt/trn1/cache/neuron-cc --auto-cast=none"
```

### 16.2 Trainium real one-layer one_hot cf4

Representative command shape:

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=32 \
  scripts/train_metis15_neuron.py \
  --manifest configs/metis15_manifest.json \
  --data-dir data/metis15_base \
  --out-dir /mnt/trn1/checkpoints/realdata_seq1024_1layer_b1g1_onehot_cf4 \
  --device xla \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 3 \
  --warmup-steps 1 \
  --constant-lr \
  --lr 1e-5 \
  --weight-decay 0.1 \
  --log-interval 1 \
  --checkpoint-interval 1000000 \
  --skip-checkpoint \
  --block-size 1024 \
  --n-layer 1 \
  --dispatch-pack-impl one_hot \
  --expert-capacity-factor 4.0
```

### 16.3 Trainium gather cf4

```bash
export NEURON_CC_FLAGS="--cache_dir=/mnt/trn1/cache/neuron-cc --auto-cast=none"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=32 \
  scripts/train_metis15_neuron.py \
  --manifest configs/metis15_manifest.json \
  --data-dir data/metis15_base \
  --out-dir /mnt/trn1/checkpoints/realdata_seq1024_1layer_b1g1_gather_cf4 \
  --device xla \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1 \
  --warmup-steps 1 \
  --constant-lr \
  --lr 1e-5 \
  --weight-decay 0.1 \
  --log-interval 1 \
  --checkpoint-interval 1000000 \
  --skip-checkpoint \
  --block-size 1024 \
  --n-layer 1 \
  --dispatch-pack-impl gather \
  --expert-capacity-factor 4.0
```

With optlevel:

```bash
export NEURON_CC_FLAGS="--cache_dir=/mnt/trn1/cache/neuron-cc --auto-cast=none --optlevel=1"
```

Still failed with graph-size limit.

### 16.4 Trainium one_hot_gather cf4

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=32 \
  scripts/train_metis15_neuron.py \
  --manifest configs/metis15_manifest.json \
  --data-dir data/metis15_base \
  --out-dir /mnt/trn1/checkpoints/realdata_seq1024_1layer_b1g1_onehot_gather_cf4 \
  --device xla \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 1 \
  --warmup-steps 1 \
  --constant-lr \
  --lr 1e-5 \
  --weight-decay 0.1 \
  --log-interval 1 \
  --checkpoint-interval 1000000 \
  --skip-checkpoint \
  --block-size 1024 \
  --n-layer 1 \
  --dispatch-pack-impl one_hot_gather \
  --expert-capacity-factor 4.0
```

Failed with `NCC_ITCT901`.

### 16.5 Trainium one_hot cf8

```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=32 \
  scripts/train_metis15_neuron.py \
  --manifest configs/metis15_manifest.json \
  --data-dir data/metis15_base \
  --out-dir /mnt/trn1/checkpoints/realdata_seq1024_1layer_b1g1_onehot_cf8 \
  --device xla \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --max-steps 3 \
  --warmup-steps 1 \
  --constant-lr \
  --lr 1e-5 \
  --weight-decay 0.1 \
  --log-interval 1 \
  --checkpoint-interval 1000000 \
  --skip-checkpoint \
  --block-size 1024 \
  --n-layer 1 \
  --dispatch-pack-impl one_hot \
  --expert-capacity-factor 8.0
```

Status: compile did not complete before interruption.

### 16.6 H100 current safe command

From H100 safe handoff:

```bash
METIS15_MOE_BACKEND=torch_grouped_safe \
METIS15_MOE_TORCH_GROUPED_MIN_M=8 \
METIS15_BATCH_SIZE=8 \
METIS15_MAX_STEPS=100 \
METIS15_LR=1e-5 \
METIS15_LM_LOSS_IMPL=standard \
METIS15_RETAIN_STANDARD_CE_LOGITS=1 \
METIS_TORCH_GROUPED_SAFE_SYNC=1 \
METIS_ASYNC_METRICS=0 \
bash scripts/metis15_h100_benchmark.sh
```

### 16.7 RTX stable command shape

Representative RTX stable command shape from Blackwell handoff:

```bash
python3 scripts/train_mamba_lm.py \
  --manifest configs/metis15_manifest.json \
  --data-dir "$METIS15_DATA_DIR" \
  --out-dir "$METIS15_OUT_DIR" \
  --train-stage pretrain \
  --batch-size 18 \
  --grad-accum-steps 11 \
  --max-steps 3 \
  --warmup-steps 2 \
  --lr 1.2e-4 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --log-interval 1 \
  --eval-interval 3 \
  --checkpoint-interval 3 \
  --dtype bf16 \
  --matmul-precision highest \
  --optimizer muon_adamw \
  --fused-adamw \
  --prefetch-batches 4 \
  --nvfp4 \
  --nvfp4-disable-rht \
  --nvfp4-disable-2d-quantization \
  --nvfp4-disable-stochastic-rounding \
  --lm-loss-impl liger_fused_linear_ce
```

## 17. What Has Already Been Tried

### GPU side

- Rebuilt RTX Blackwell Docker image on NGC PyTorch 26.04.
- Rebuilt Transformer Engine main for SM120/120a.
- Patched TE MXFP8 guard experimentally.
- Added exact-shape Blackwell kernel smoke tests.
- Added reduced NVFP4 support flags.
- Added FP8-block fallback surfaces.
- Added final-2 expert higher precision option.
- Added Liger fused linear CE to launcher path.
- Replaced old Python per-expert loop with grouped MoE dispatch.
- Added bucketed dispatch and reverse weighted combine.
- Added TE SwiGLU path in some experiments.
- Added PyTorch `_grouped_mm` backend on H100.
- Added `torch_grouped_safe` guards.
- Added no-empty expert padding.
- Added Triton fused SwiGLU experiment.
- Added pinned CUDA prefetcher.
- Added foreach AdamW in Muon-AdamW hybrid.
- Added optimizer NVTX ranges.
- Added active/rough parameter-application audit.
- Verified next-token CE alignment.
- Ran real-data H100 and RTX smoke/benchmark passes.
- Ran NSYS profile showing launch/GEMM fragmentation.

### Trainium side

- Discovered correct SSH key/user.
- Built RAID0 scratch at `/mnt/trn1`.
- Installed Neuron stack.
- Ran one-core BF16 XLA smoke.
- Added `scripts/neuron_synthetic_bench.py`.
- Ran one-core and 32-core synthetic dense/local-expert probes.
- Implemented/ran `scripts/train_metis15_neuron.py` static EP trainer.
- Tested `index_add`, `one_hot`, `gather`, `one_hot_gather`.
- Proved CPU equivalence for dispatcher variants.
- Found Neuron compiler graph-size and transform blockers.
- Got one-layer seq1024 real-data cached `one_hot` to about `37k tok/s`.

## 18. What Not To Conclude

Do not conclude:

- "Trainium is slow." Synthetic probes were very fast. The current Metis graph is the issue.
- "Expert parallelism did nothing." EP sharded routed experts, but dispatch/combine/dense/loss overhead still dominates.
- "The 5M number was real training." It was synthetic/static.
- "The one-layer 37k Trainium result is full pretraining throughput." It is not.
- "H100 is weak." Dense Metis-1.4 synthetic FP8 reached `146k-147k tok/s` on the H100 image.
- "RTX PRO 6000 cannot do low precision." Exact public TE/cuBLAS paths failed or underperformed for this workload; that is different from hardware impossibility.
- "Optimizer is the main time bottleneck." RTX profile showed optimizer time was not first-order, although optimizer memory is important.
- "Batch size alone fixes this." Batch increases helped a little on H100 then hit stability/OOM; Trainium batch increases need a compile-friendly dispatcher first.
- "NaN throughput counts." Any throughput after non-finite loss is invalid.
- "Reduced NVFP4/MXFP8 existence means usable." Exact train forward/backward shapes must pass and be faster.

## 19. High-Value Research Questions

### 19.1 Trainium-specific

1. What is the recommended NeuronX Distributed / NxD / Megatron-style implementation for top-k MoE expert parallelism on `trn1.32xlarge`?
2. Is there an official Neuron token dispatcher/all-to-all MoE path that avoids giant Torch/XLA one-hot selectors and compiler-size blowups?
3. Can `xm.all_to_all` payloads be packed with a lower-level op or custom call rather than huge graph-level `one_hot` or `argsort/searchsorted`?
4. Can the graph be split into compile-friendly pieces without killing throughput?
5. Does Neuron support efficient attention for this `d_model=1536`, `heads=24`, `seq=1024` shape, or do we need a different attention implementation?
6. Should dense components be tensor-parallel/vocab-parallel on Trainium?
7. Is `world_size=32` with one local expert per rank too fine-grained? Would `EP=8` on 32 NeuronCores plus tensor/data parallelism be better?
8. How should router load balancing be implemented so cf4 or lower does not drop assignments?
9. Is 500k tok/s realistic on a single `trn1.32xlarge` for this model after all overheads?
10. If not, what is the best realistic sustained tok/s on trn1.32xlarge, and what hardware topology gets to 500k?

### 19.2 GPU-specific

1. What fused MoE kernel path should replace current grouped GEMM + sort/gather/scatter?
2. Is PyTorch `_grouped_mm` the wrong abstraction for H100/Metis expert shapes?
3. Is CUTLASS grouped GEMM or cuDNN Frontend grouped GEMM viable for this exact top-k MoE path?
4. Can SwiGLU forward/backward be safely fused for routed experts?
5. Can static capacity be made semantically correct without padding away throughput?
6. Is FP8/NVFP4 realistically recoverable for current shapes and current toolchains?
7. Should the model shape change to improve kernel alignment, or should implementation pad internally?
8. Can FlashAttention-3 or another attention backend materially affect total throughput once MoE is fixed?

### 19.3 Architecture/accounting

1. Is the current `param_application_audit()` accurate for the current single-latent manifest?
2. What useful TFLOP/s is required for `500k tok/s` under the true training FLOP accounting?
3. Does top-k=4 make the model too expensive for the target without much stronger fused kernels?
4. Is the shared expert cost too high relative to routed experts?
5. Would reducing top-k, changing expert hidden size, or changing latent dim preserve model goals while making throughput feasible?
6. Which changes are implementation-only versus architecture-changing?

## 20. Candidate Routes To 500k tok/s

These are candidate directions, not conclusions.

### Route A: Real Trainium/NxD MoE stack

Port Metis-1.5 to an official NeuronX Distributed or Megatron-compatible stack:

- expert parallel token dispatcher
- tensor parallel dense projections
- sharded/vocab-parallel lm head and CE
- distributed optimizer
- compiler-friendly static shapes
- supported attention implementation
- avoid giant Python/Torch graph for dispatch

This is probably the most credible Trainium route.

### Route B: Custom fused GPU MoE path

For H100/RTX:

- counting-sort/token histogram dispatcher for 32 experts
- persistent buffers
- grouped FC1 + activation fusion
- grouped FC2 + gate/weight fusion
- reverse combine without atomics where possible
- safe backward/Wgrad/Dgrad
- graph-capturable fixed shape

This matches the RTX profile evidence.

### Route C: Larger batch after dispatcher fix

After dispatch is fixed:

- sweep local batch on Trainium
- sweep microbatch/grad accumulation on H100/RTX
- measure true steady-state after compile/warmup
- ensure optimizer state is materialized before declaring throughput

Current batch is too small on Trainium, but increasing it before fixing dispatch may just increase compile pain.

### Route D: Dense/loss parallelism

Shard:

- qkv/o projections
- lm head
- cross entropy
- optimizer state

EP alone is likely insufficient because dense replicated work remains too large.

### Route E: Shape/architecture adjustment

Only if implementation paths cannot reach target:

- reduce top-k from 4 to 2
- adjust expert hidden size
- adjust latent dim
- modify shared expert
- change number of experts/local experts

These are architecture changes and should be treated carefully.

## 21. Verification Checklist For Any Proposed Fix

Any proposed solution should be judged by this checklist:

1. Does it run real `metis15_base` data, not synthetic only?
2. Does it run at `seq_len=1024`?
3. Does it run the current single-latent manifest shape unless explicitly proposing architecture change?
4. Does it include forward, backward, optimizer step, and loss?
5. Does it avoid dropped/overflowed assignments, or does it prove overflow is semantically safe?
6. Does it survive at least 100 steady-state steps without NaN?
7. Does it ignore warmup/compile time when reporting steady-state, while still reporting compile time separately?
8. Does it log median/p50/p95 step time, not just final line?
9. Does it prove optimizer state is already materialized?
10. Does it state whether checkpoints/eval are included or excluded?
11. Does it report global tokens/sec and tokens per optimizer step?
12. Does it report hardware utilization/profiler breakdown?
13. Does it compare against the same baseline shape?
14. Does it preserve model semantics, or clearly label architecture changes?
15. Does it project ETA for 50B tokens with overhead?

## 22. Appendix: File Map

Primary current files:

```text
configs/metis15_manifest.json
scripts/train_metis15_neuron.py
scripts/metis15_neuron_pretrain.sh
scripts/neuron_synthetic_bench.py
scripts/train_mamba_lm.py
scripts/metis15_h100_benchmark.sh
scripts/metis15_rtx_benchmark_matrix.sh
src/metis_mamba/config.py
src/metis_mamba/model.py
src/metis_mamba/moe_kernels.py
src/metis_mamba/optim.py
Makefile
```

Key docs already in repo:

```text
docs/metis15_rtx_pro_6000_fp8_throughput_research_packet_2026-05-14.md
docs/metis15_rtx_pro_6000_blackwell_testing_handoff_2026-05-14.md
docs/metis15_throughput_patch_2026-05-14.md
docs/metis15_h100_torch_grouped_moe_findings_2026-05-14.md
docs/metis15_h100_single_latent_testing_handoff_2026-05-15.md
docs/metis15_h100_torch_grouped_safe_followup_2026-05-15.md
docs/metis15_h100_fused_moe_kernel_attempt_2026-05-14.md
docs/metis15_8xa100_expert_parallel.md
docs/metis15_nemotron_megatron_research_2026-05-17.md
```

Make targets:

```text
make metis15-neuron-pretrain
make metis15-a100-pretrain
make metis15-rtx-benchmark-matrix
make metis15-megatron-profile
```

Important unresolved artifact:

```text
/mnt/trn1/checkpoints/realdata_seq1024_1layer_b1g1_onehot_cf8/run.log
```

This log could not be refreshed because the current Trainium host timed out over SSH.

## Final Bottom Line For The Research Model

The core mystery is not "why is Trainium only 37k?" It is:

```text
Why does the real Metis-1.5 MoE training dataflow collapse very different hardware
into the same 30k-40k tok/s class, while synthetic/dense/static probes can go far higher?
```

My current best answer is:

```text
The implementation is dominated by fragmented MoE routing/dispatch/combine, small/ragged expert GEMMs,
graph/compiler overhead, unfused activations, and replicated dense/loss work. Expert parallelism alone
reduced expert ownership but did not replace the expensive token dispatcher or shard the dense path.
```

The path to `500k tok/s` likely requires one or both of:

1. A real production MoE stack on Trainium using NeuronX Distributed/Megatron-style tensor+expert parallelism and compiler-friendly token dispatch.
2. A fused custom GPU MoE path that removes the launch/sort/gather/scatter fragmentation seen in the RTX/H100 profiles.

Small flag sweeps around the current raw Torch/XLA `one_hot` dispatcher are unlikely to produce a `10x-15x` throughput jump.
