FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG TORCHTITAN_COMMIT=08da451f2604d6c277e8b206de223c86fdb7935a
ARG TORCHAO_COMMIT=b3e0db2fae37427b867f8d2b43a0d94d1a474249
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TORCHTITAN_DIR=/opt/torchtitan
ENV METIS_IMAGE_ROOT=/opt/metis
ENV VIRTUAL_ENV=/opt/metis-venv
ENV PATH=/opt/metis-venv/bin:$PATH

WORKDIR /opt

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
      git \
      curl \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

COPY requirements-gpu-train.txt /tmp/requirements-gpu-train.txt
RUN printf '%s\n' \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    > /tmp/torch-constraints.txt

RUN PIP_CONSTRAINT=/tmp/torch-constraints.txt \
    pip install --upgrade pip "setuptools<82" wheel && \
    pip install \
      --index-url https://download.pytorch.org/whl/cu128 \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" && \
    pip install -r /tmp/requirements-gpu-train.txt

RUN git clone https://github.com/pytorch/torchtitan "$TORCHTITAN_DIR" && \
    git -C "$TORCHTITAN_DIR" checkout "$TORCHTITAN_COMMIT" && \
    PIP_CONSTRAINT=/tmp/torch-constraints.txt pip install -r "$TORCHTITAN_DIR/requirements.txt" && \
    PIP_CONSTRAINT=/tmp/torch-constraints.txt pip install -e "$TORCHTITAN_DIR"

RUN PIP_CONSTRAINT=/tmp/torch-constraints.txt \
    USE_CPP=0 pip install --no-build-isolation \
    "git+https://github.com/pytorch/ao.git@${TORCHAO_COMMIT}"

WORKDIR $METIS_IMAGE_ROOT
COPY . $METIS_IMAGE_ROOT
RUN PIP_CONSTRAINT=/tmp/torch-constraints.txt pip install -e . && \
    python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda)"

WORKDIR /workspace
