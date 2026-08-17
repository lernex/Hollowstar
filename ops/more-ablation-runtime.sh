# MoRE ablation runtime activation. Sourced by every generated sbatch via
# METIS_ABLATION_RUNTIME, once per task, so the per-rank scratch paths below
# resolve against that task's SLURM_PROCID.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cray-python/3.11.7
module load rocm/7.2.1
export ROCM_PATH=/opt/rocm-7.2.1
export HIP_HOME=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx942
# Composable Kernel supplies the grouped GEMM the replicated-expert bank runs
# on. Without it Transformer Engine falls back to a path that is not merely
# slower but unusable at 96 experts -- five calls did not finish in nine
# minutes with the GPU pegged, against 1.75 ms each on CK.
export NVTE_USE_CK_GROUPED_GEMM=1
# PyTorch defaults to hipBLASLt for aten GEMMs on gfx942, and on this build
# hipBLASLt answers essentially every shape this model asks for with about
# 490 ms of host time for microseconds of matrix-core work. rocBLAS does the
# same GEMMs in 0.15 to 0.54 ms. That single default is where the 0.06% MFU
# came from. Transformer Engine keeps its own hipBLASLt path for FP8; this
# only redirects the aten surface.
export TORCH_BLAS_PREFER_HIPBLASLT=0
# MI300A takes its HBM from system memory, so allocator fragmentation is more
# expensive here than on a discrete part: a fragmented pool cannot be papered
# over by spare VRAM. Expandable segments grow one virtual reservation instead
# of scattering fixed blocks (PyTorch HIP notes, 2026).
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
# RCCL on Slingshot: let the library pin ranks itself, give it enough channels
# for four APUs per node, and allow GPU-Direct RDMA (RCCL usage tips, 2026).
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_MIN_NCHANNELS=24
export NCCL_NET_GDR_LEVEL=1
# flash_attn resolves its ROCm backend at import time; name it rather than
# relying on the auto-detect branch.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# AITER autotunes the attention backward at first call, compiling dozens of
# configs concurrently; that is where the compiler aborts with
# "LLVM ERROR: IO failure on output stream". It is also the wrong thing for
# this campaign even when it works: each row would select its own kernels, and
# wall clock is one of the axes the campaign reports.
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0
# Triton JIT-compiles and writes each kernel here. Four ranks per node sharing
# one directory corrupt each other's output; it surfaces as
# LLVM ERROR: IO failure on output stream, which names neither Triton nor the
# cache. Keep one per rank, on node-local disk rather than Lustre.
# Node-local NVMe, not /tmp.  /tmp on these nodes is a tmpfs, and an MI300A
# APU takes its HBM from the same system memory the tmpfs lives in -- a
# four-rank node runs at roughly 385 GB of 501 GB before the compiler writes a
# byte.  Triton spilling its output into that is where
# "LLVM ERROR: IO failure on output stream: Bad address" comes from; the error
# names neither memory nor the cache.  /data is a 1.8 TB local disk.
METIS_NODE_SCRATCH=/data/$USER/metis-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}
export TRITON_CACHE_DIR=$METIS_NODE_SCRATCH/triton
export MIOPEN_USER_DB_PATH=$METIS_NODE_SCRATCH/miopen
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export HIP_FORCE_DEV_KERNARG=1
# AITER JIT-builds into ~/.aiter by default -- shared NFS home, one directory
# for every rank on every node. Twenty-eight ranks racing a copytree and a
# build there is what produced LLVM ERROR: IO failure on output stream.
# Triton's AMD backend also links through NamedTemporaryFile in TMPDIR.
# Both go per rank, on node-local tmpfs.
export TMPDIR=$METIS_NODE_SCRATCH/tmp
export AITER_JIT_DIR=$TMPDIR/aiter
mkdir -p $TRITON_CACHE_DIR $MIOPEN_USER_DB_PATH $TMPDIR $AITER_JIT_DIR
source /lus/lustre1/vollmerc/more-runtime/venv311/bin/activate
