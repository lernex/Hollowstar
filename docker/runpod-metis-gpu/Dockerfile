FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG TORCH_VERSION=2.8.0
ARG TRANSFORMER_ENGINE_REF="v2.15"
ARG FLASH_ATTENTION_REPO="https://github.com/Dao-AILab/flash-attention.git"
ARG FLASH_ATTENTION_REF="v2.8.3"
ARG INSTALL_FLASH_ATTENTION=0
ARG INSTALL_TE_FUSED_GROUPED_MLP_DEPS=0
ARG TE_BUILD_MAX_JOBS=8

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    METIS_IMAGE_ROOT=/opt/metis \
    VIRTUAL_ENV=/opt/metis-venv \
    PATH=/opt/metis-venv/bin:$PATH \
    METIS13_PYTHON_BIN=/opt/metis-venv/bin/python \
    METIS13_SKIP_ENV_SETUP=1 \
    METIS15_PYTHON_BIN=/opt/metis-venv/bin/python \
    METIS15_SKIP_ENV_SETUP=1 \
    TORCH_CUDA_ARCH_LIST=12.0a \
    NVTE_CUDA_ARCHS=120a \
    MAX_JOBS=$TE_BUILD_MAX_JOBS \
    NVTE_BUILD_MAX_JOBS=$TE_BUILD_MAX_JOBS \
    NVCC_THREADS=1 \
    NVTE_BUILD_THREADS_PER_JOB=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=$TE_BUILD_MAX_JOBS \
    CUDA_PATH=/usr/local/cuda

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      build-essential \
      ca-certificates \
      git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages "$VIRTUAL_ENV"

COPY requirements-gpu-train.txt /tmp/requirements-gpu-train.txt
RUN pip install --upgrade pip "setuptools<82" wheel ninja && \
    pip install -r /tmp/requirements-gpu-train.txt

RUN pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-torch \
      transformer_engine transformer_engine_cu12 transformer_engine_torch >/dev/null 2>&1 || true && \
    pip install pybind11 "nvidia-mathdx==25.6.0" && \
    if [[ "$INSTALL_TE_FUSED_GROUPED_MLP_DEPS" == "1" ]]; then \
      pip uninstall -y cutlass nvidia-cutlass nvidia-cutlass-dsl \
        nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 nvidia-cudnn-frontend || true && \
      pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.4.1" && \
      pip install --no-cache-dir "nvidia-cudnn-frontend[cutedsl]==1.23.0"; \
    fi && \
    git clone --branch "$TRANSFORMER_ENGINE_REF" --recursive --depth 1 https://github.com/NVIDIA/TransformerEngine.git /tmp/TransformerEngine && \
    cd /tmp/TransformerEngine && \
    NVTE_FRAMEWORK=pytorch pip install --no-build-isolation . && \
    rm -rf /tmp/TransformerEngine

ENV FLASH_ATTENTION_DISABLE_SM80=TRUE \
    FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
    FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
    FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
    FLASH_ATTENTION_DISABLE_LOCAL=TRUE \
    FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
    FLASH_ATTENTION_DISABLE_PACKGQA=TRUE \
    FLASH_ATTENTION_DISABLE_FP16=TRUE \
    FLASH_ATTENTION_DISABLE_FP8=TRUE \
    FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
    FLASH_ATTENTION_DISABLE_CLUSTER=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM128=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM256=TRUE

RUN if [[ "$INSTALL_FLASH_ATTENTION" == "1" ]]; then \
      git clone --branch "$FLASH_ATTENTION_REF" --depth 1 "$FLASH_ATTENTION_REPO" /tmp/flash-attention && \
      cd /tmp/flash-attention/hopper && \
      python setup.py install && \
      rm -rf /tmp/flash-attention; \
    else \
      echo "Skipping Hopper-only FlashAttention build for the default RTX PRO 6000 / sm120 image."; \
    fi

RUN python - <<'PY'
from pathlib import Path
import site

site_packages = Path(site.getsitepackages()[0])
try:
    import flash_attn_interface  # noqa: F401
except ImportError:
    print("flash_attn_interface not installed; SDPA/Transformer Engine attention remains the default.")
else:
    flash_attn_pkg = site_packages / "flash_attn_3"
    flash_attn_pkg.mkdir(exist_ok=True)
    (flash_attn_pkg / "__init__.py").write_text("from .flash_attn_interface import *\n", encoding="utf-8")
    (flash_attn_pkg / "flash_attn_interface.py").write_text(
        "from flash_attn_interface import *\n"
        "from flash_attn_interface import _flash_attn_backward, _flash_attn_forward\n"
        "from flash_attn_interface import flash_attn_func, flash_attn_varlen_func\n"
        "from flash_attn_interface import flash_attn_with_kvcache\n",
        encoding="utf-8",
    )
PY

COPY . "$METIS_IMAGE_ROOT"

WORKDIR $METIS_IMAGE_ROOT
RUN pip install -e . && \
    python - <<'PY'
import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

try:
    import flash_attn_interface as flash_attn_mod
    flash_attn_source = "flash_attn_interface"
except ImportError:
    try:
        from hopper import flash_attn_interface as flash_attn_mod
        flash_attn_source = "hopper.flash_attn_interface"
    except ImportError:
        flash_attn_mod = None
        flash_attn_source = "not-installed"

assert torch.__version__.startswith("2.8.0"), torch.__version__
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformer_engine", getattr(te, "__file__", "missing"))
print("transformer_engine_nvfp4", hasattr(recipe, "NVFP4BlockScaling"))
print("transformer_engine_mxfp8", hasattr(recipe, "MXFP8BlockScaling"))
print("flash_attention", flash_attn_source, getattr(flash_attn_mod, "__file__", "not-installed"))
print("liger", LigerFusedLinearCrossEntropyLoss)
PY

RUN python scripts/train_mamba_lm.py --help >/dev/null && \
    python scripts/train_mamba_sft.py --help >/dev/null && \
    python scripts/train_mamba_reward.py --help >/dev/null && \
    python scripts/train_mamba_dpo.py --help >/dev/null

WORKDIR /opt/metis
