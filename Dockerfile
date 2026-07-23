# syntax = docker/dockerfile:experimental@sha256:3c244c0c6fc9d6aa3ddb73af4264b3a23597523ac553294218c13735a2c6cf79
ARG UBUNTU_VERSION=24.04

ARG ARCH=
ARG CUDA=12.9.1
ARG CUDA_SHORT=12.9
ARG CUDA_PACKAGE_VERSION=12-9
ARG CUDA_FLAVOR=base
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-${CUDA_FLAVOR}-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDA_SHORT
ARG CUDA_PACKAGE_VERSION
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python, pip, and dos2unix for Windows-friendly script execution.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip dos2unix && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA CUDA/cuDNN runtime libraries needed by TensorFlow on x86.
COPY dependencies/install_cuda.sh ./install_cuda.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    dos2unix ./install_cuda.sh && \
    /bin/bash ./install_cuda.sh && \
    rm install_cuda.sh && \
    rm -rf /var/lib/apt/lists/*

# Copy Python requirements in and install them (--break-system-packages is required if we don't use a venv)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-compile -r requirements.txt

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
