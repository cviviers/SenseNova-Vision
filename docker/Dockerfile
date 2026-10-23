# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SenseNova-Vision runtime for NVIDIA Blackwell GPUs (RTX 50xx / sm_120, B200 / sm_100).
#
# The reference environment (PyTorch 2.5.1 + CUDA 12.4) ships no sm_120 kernels,
# so this image moves to PyTorch 2.8 + CUDA 12.8 and a flash-attn wheel built for
# that stack. Every other pin in requirements.txt / constraints.txt is kept.
#
# Build from the repository root:
#   docker compose -f docker/docker-compose.yml build

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="10.0;12.0" \
    JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Python 3.10 matches environment.yml; OpenJDK 8 is needed by pycocoevalcap.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        build-essential git curl wget ca-certificates \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 ffmpeg \
        openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv ${VIRTUAL_ENV} \
    && pip install --upgrade pip==25.1 setuptools==78.1.1 wheel==0.45.1

RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url ${TORCH_INDEX_URL}

# Reuse the repository pins, dropping only the entries tied to the CUDA 12.4
# reference stack (torch, torchvision, triton, nvidia-* runtime wheels) and
# sympy, which torch 2.8 requires at >=1.13.3.
COPY requirements.txt constraints.txt docker/requirements-blackwell.txt /tmp/sensenova/
RUN cd /tmp/sensenova \
    && grep -v -E '^(torch|torchvision|triton)==' requirements.txt > requirements.docker.txt \
    && grep -v -E '^(nvidia-[a-z0-9-]+|sympy)==' constraints.txt > constraints.docker.txt \
    && pip install -r requirements.docker.txt -r requirements-blackwell.txt \
        -c constraints.docker.txt \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --extra-index-url ${TORCH_INDEX_URL}

RUN pip install "${FLASH_ATTN_WHEEL}"

# Evaluation extensions (see setup.sh).
COPY tools/evaluation/detect/evaluation/fastevaluate /opt/sensenova/fastevaluate
COPY tools/evaluation/recons/requirements.txt /tmp/sensenova/recons-requirements.txt
RUN pip install "panopticapi @ https://github.com/cocodataset/panopticapi/archive/7bb4655548f98f3fedc07bf37e9040a992b054b0.zip" \
        -c /tmp/sensenova/constraints.docker.txt \
    && pip install /opt/sensenova/fastevaluate \
    && grep -v -E '^(torch|torchvision)==' /tmp/sensenova/recons-requirements.txt > /tmp/sensenova/recons.docker.txt \
    && pip install -r /tmp/sensenova/recons.docker.txt \
        -c /tmp/sensenova/constraints.docker.txt \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --extra-index-url ${TORCH_INDEX_URL}

# Generation metrics: clean-fid (FID/KID with the standard Inception weights) and timm (DINOv2 features).
# --no-deps keeps every pin above unchanged; their dependencies are already installed.
RUN pip install --no-deps clean-fid==0.1.35 timm==1.0.30

COPY docker/check_env.py /opt/sensenova/check_env.py
COPY docker/entrypoint.sh /usr/local/bin/sensenova-entrypoint
RUN chmod +x /usr/local/bin/sensenova-entrypoint

# The repository itself is bind-mounted here by docker-compose, so code edits
# on the host take effect without rebuilding.
WORKDIR /workspace/SenseNova-Vision
ENV PYTHONPATH=/workspace/SenseNova-Vision:/workspace/SenseNova-Vision/inference \
    MODEL_PATH=/models/SenseNova-Vision-7B-MoT \
    SENSENOVA_MODEL_DIR=/models/SenseNova-Vision-7B-MoT \
    HF_HOME=/hf-cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 9001
ENTRYPOINT ["sensenova-entrypoint"]
CMD ["bash"]
