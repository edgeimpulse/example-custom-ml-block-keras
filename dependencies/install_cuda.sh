#!/bin/bash
set -ex

UNAME=$(uname -m)

if [ "$UNAME" == "aarch64" ]; then
    echo "Skipping CUDA on AARCH64..."
else
    CUDA_PKG_VERSION=${CUDA_PKG_VERSION:?CUDA_PKG_VERSION is required}
    CUDNN_PACKAGE=${CUDNN_PACKAGE:?CUDNN_PACKAGE is required}

    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA_PKG_VERSION} \
        cuda-nvcc-${CUDA_PKG_VERSION} \
        libcublas-${CUDA_PKG_VERSION} \
        cuda-nvrtc-${CUDA_PKG_VERSION} \
        libcufft-${CUDA_PKG_VERSION} \
        libcurand-${CUDA_PKG_VERSION} \
        libcusolver-${CUDA_PKG_VERSION} \
        libcusparse-${CUDA_PKG_VERSION} \
        libnvjitlink-${CUDA_PKG_VERSION} \
        curl \
        ${CUDNN_PACKAGE} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

    apt-get clean \
        && rm -rf /var/lib/apt/lists/*

    if [ -f /usr/local/cuda/lib64/stubs/libcuda.so ]; then
        ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
            && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
            && ldconfig
    fi
fi
