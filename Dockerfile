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

# CUDA drivers
SHELL ["/bin/bash", "-c"]
COPY dependencies/install_cuda.sh ./install_cuda.sh
RUN ./install_cuda.sh && \
    rm install_cuda.sh

# Install base packages (like Python and pip)
RUN apt update && apt install -y curl zip git lsb-release software-properties-common apt-transport-https vim wget python3 python3-pip
RUN python3 -m pip install --upgrade pip==20.3.4

COPY dependencies/akida ./akida
RUN cd akida && ./download_akida.sh
RUN pip3 install --no-cache-dir -r akida/requirements.txt

# Patch cnn2snn library for Keras 2.11 compat
RUN cd akida && /bin/bash ./patch-cnn2snn.sh && \
    rm patch-cnn2snn.sh

# Copy Python requirements in and install them
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

COPY transfer-learning-weights ./transfer-learning-weights
# Download the BrainChip Akidanet transfer learning base weights from CDN
RUN apt-get install -y parallel
RUN cd transfer-learning-weights/akidanet && \
    cat weights.txt | parallel -j 10 --gnu "wget http://cdn.edgeimpulse.com/transfer-learning-weights/akidanet/{}"

WORKDIR /scripts

# Copy the rest of your training scripts in
COPY . ./


# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
