#!/usr/bin/env bash
set -euo pipefail

architecture="$(dpkg --print-architecture)"
if [[ "${architecture}" != "amd64" ]]; then
    echo "Skipping NVIDIA CUDA/cuDNN package install on ${architecture}"
    exit 0
fi

cuda_short="${CUDA_SHORT:-12.9}"
cuda_package_version="${CUDA_PACKAGE_VERSION:-12-9}"

apt-get update
apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 libcublas-${cuda_package_version} \
    libcufft-${cuda_package_version} libcurand-${cuda_package_version} \
    libcusolver-${cuda_package_version} libcusparse-${cuda_package_version} \
    cuda-nvrtc-${cuda_package_version} libnvjitlink-${cuda_package_version}

apt-get download cuda-nvvm-${cuda_package_version} cuda-nvcc-${cuda_package_version}
mkdir -p /tmp/cuda-nvvm /tmp/cuda-nvcc \
    /usr/local/cuda-${cuda_short}/bin \
    /usr/local/cuda-${cuda_short}/nvvm/libdevice
dpkg-deb -x cuda-nvvm-${cuda_package_version}_*.deb /tmp/cuda-nvvm
dpkg-deb -x cuda-nvcc-${cuda_package_version}_*.deb /tmp/cuda-nvcc
cp /tmp/cuda-nvcc/usr/local/cuda-${cuda_short}/bin/ptxas \
    /usr/local/cuda-${cuda_short}/bin/
cp /tmp/cuda-nvvm/usr/local/cuda-${cuda_short}/nvvm/libdevice/libdevice.10.bc \
    /usr/local/cuda-${cuda_short}/nvvm/libdevice/

rm -f /usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so*
rm -rf /tmp/cuda-nvvm /tmp/cuda-nvcc \
    cuda-nvvm-${cuda_package_version}_*.deb cuda-nvcc-${cuda_package_version}_*.deb \
    /usr/share/doc/* /usr/share/man/* /usr/share/locale/*