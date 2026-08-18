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
# RCCL on Slingshot: let the library pin ranks itself and give it enough
# channels for four APUs per node.
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_MIN_NCHANNELS=24

# The in-tree ROCm wheel otherwise falls back to NET/Socket. Measured on
# Portage on 2026-08-17, a 1 GiB two-node all-reduce moved from 2.49 GB/s over
# sockets to 52.3 GB/s over OFI/CXI GDRDMA. HPE's June 2026 Slingshot RCCL
# guidance requires both the OFI plugin and these CXI rendezvous settings.
# Do not force OFI for a single-node step: it needlessly allocates a VNI and is
# slower than the native intra-node path.
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
  METIS_RCCL_OFI_ROOT=${METIS_RCCL_OFI_ROOT:-/lus/lustre1/vollmerc/more-runtime/rccl-ofi-v1.21.0-rocm7.2.1-pcihwloc/opt}
  METIS_RCCL_OFI_PLUGIN=$METIS_RCCL_OFI_ROOT/aws-ofi-rccl/lib/librccl-net.so
  if [ ! -s "$METIS_RCCL_OFI_PLUGIN" ]; then
    echo "metis: missing required Slingshot RCCL plugin: $METIS_RCCL_OFI_PLUGIN" >&2
    return 1
  fi
  export LD_LIBRARY_PATH="$METIS_RCCL_OFI_ROOT/aws-ofi-rccl/lib:$METIS_RCCL_OFI_ROOT/hwloc/lib:${LD_LIBRARY_PATH:-}"
  export NCCL_NET_PLUGIN=$METIS_RCCL_OFI_PLUGIN
  export NCCL_NET=OFI
  export NCCL_CROSS_NIC=1
  export NCCL_NET_GDR_LEVEL=PHB
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export HSA_FORCE_FINE_GRAIN_PCIE=1
  export FI_PROVIDER=cxi
  export FI_MR_CACHE_MONITOR=userfaultfd
  export FI_CXI_DISABLE_HOST_REGISTER=1
  export FI_CXI_DEFAULT_CQ_SIZE=131072
  export FI_CXI_RDZV_PROTO=alt_read
  export FI_CXI_RX_MATCH_MODE=hybrid
  export FI_CXI_RDZV_EAGER_SIZE=0
  export FI_CXI_RDZV_THRESHOLD=0
  export FI_CXI_RDZV_GET_MIN=0
  export FI_CXI_DEFAULT_TX_SIZE=2048
fi
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
# /data is a per-node disk and it is not healthy on every node: parrypeak061
# answers mkdir with an I/O error while its four APUs are perfectly fine, and a
# job that assumes otherwise loses the whole row to one bad mount. Fall back to
# the tmpfs rather than the run.
if ! mkdir -p "$METIS_NODE_SCRATCH" 2>/dev/null; then
  METIS_NODE_SCRATCH=/tmp/metis-$USER-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}
  echo "metis: /data unusable on $(hostname), falling back to $METIS_NODE_SCRATCH" >&2
fi
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
# AITER JIT-compiles its core module on first import. Twenty-eight ranks each
# running a full C++ build at once, four to a node that shares one memory pool
# with its APUs, segfaulted a node outright and cost minutes of every launch.
# The module is a few tens of kilobytes; build it once and copy it in.
METIS_AITER_PREBUILT=/lus/lustre1/vollmerc/more-runtime/aiter-prebuilt
if [ -d "$METIS_AITER_PREBUILT" ]; then
  cp -r "$METIS_AITER_PREBUILT"/. "$AITER_JIT_DIR"/ 2>/dev/null || true
fi
source /lus/lustre1/vollmerc/more-runtime/venv311/bin/activate
