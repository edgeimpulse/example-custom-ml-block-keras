# syntax = docker/dockerfile:experimental@sha256:3c244c0c6fc9d6aa3ddb73af4264b3a23597523ac553294218c13735a2c6cf79
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.2
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.2-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDNN=8.1.0.77-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# When building on Windows we'll get CRLF line endings, which we cannot run from bash...
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt update && apt install -y dos2unix && \
    rm -r /var/lib/apt/lists/

# CUDA drivers
SHELL ["/bin/bash", "-c"]
COPY dependencies/install_cuda.sh ./install_cuda.sh
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    dos2unix ./install_cuda.sh && \
    /bin/bash ./install_cuda.sh && \
    rm install_cuda.sh && \
    rm -r /var/lib/apt/lists/

# Install base packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt update && apt install -y curl zip git lsb-release software-properties-common apt-transport-https vim wget && \
    rm -r /var/lib/apt/lists/

# Install Python 3
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt update && apt install -y python3 python3-pip python3-distutils && \
    rm -r /var/lib/apt/lists/

# Pin pip / setuptools
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install "pip==21.3.1" "setuptools==62.6.0"

# Copy Python requirements in and install them
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
