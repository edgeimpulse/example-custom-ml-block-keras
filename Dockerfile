# syntax = docker/dockerfile:experimental@sha256:3c244c0c6fc9d6aa3ddb73af4264b3a23597523ac553294218c13735a2c6cf79
ARG UBUNTU_VERSION=22.04

ARG ARCH=
ARG CUDA=12.9
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.0-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDNN=9.10.2.21-1
ARG CUDNN_MAJOR_VERSION=9
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=10.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=10
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# CUDA drivers
SHELL ["/bin/bash", "-c"]
COPY dependencies/install_cuda.sh ./install_cuda.sh
RUN /bin/bash ./install_cuda.sh && \
    rm install_cuda.sh

# Install base packages
RUN apt update && apt install -y curl zip git lsb-release software-properties-common apt-transport-https vim wget

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-distutils
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip install --upgrade pip==20.3.4

# Copy Python requirements in and install them
COPY requirements.txt ./
RUN pip3.10 install -r requirements.txt

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3.10", "-u", "train.py"]
