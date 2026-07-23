# Plan: Upgrade Docker GPU Stack

Upgrade this repository from Ubuntu 20.04 / CUDA 11.2 / TensorFlow 2.11 to an Ubuntu 24.04 Docker image that runs TensorFlow 2.21 on an RTX 4090. Keep `train.py` essentially unchanged, keep the existing `nvidia/cuda` base-image structure, and validate with the exact GPU training command from `GOALS.md`.

## Steps

1. Establish the expected output.
   - Preserve `Training on: gpu` from real TensorFlow GPU detection.
   - Preserve real NVIDIA/TensorFlow runtime evidence that the RTX 4090 is visible.
   - Preserve validation accuracy around 0.98 after 5 epochs.
   - Preserve `Saving saved model OK` and `out/saved_model.zip`.

2. Choose and test the upgraded NVIDIA-owned `nvidia/cuda` Ubuntu 24.04 base tag.
   - Keep the Dockerfile pattern that starts from `nvidia/cuda${ARCH:+-$ARCH}:...-ubuntu...`.
   - Use NVIDIA's smaller CUDA/cuDNN runtime flavor and install only the CUDA package that proved necessary for TensorFlow GPU JIT compilation.
   - Install `cuda-nvvm-12-9` for `libdevice.10.bc`; leave out `cuda-nvcc-12-9`/`ptxas` to reduce image size, accepting TensorFlow's driver-compiler fallback warnings.
   - Do not switch to a TensorFlow-provided image, generic Python image, or any other non-`nvidia/cuda` base image.

3. Remove the custom CUDA dependency installation path.
   - Skip `dependencies/install_cuda.sh` because the official NVIDIA `cudnn-runtime` image plus `cuda-nvvm-12-9` supplies the required CUDA/cuDNN runtime and `libdevice` file.
   - Preserve the Docker flow after the base image: minimal Python/pip install, requirements, app copy, and training entrypoint.

4. Update Python dependencies for TensorFlow 2.21.
   - Install TensorFlow 2.21.
   - Remove stale TensorFlow 2.11-era pins such as `keras==2.11.0` and `protobuf==3.19.*` unless validation proves a new explicit pin is required.
   - Use a NumPy version accepted by TensorFlow 2.21 and the Python version in Ubuntu 24.04.
   - Install only `python3` and `python3-pip` from apt; `python3-pip` pulls `setuptools`, `wheel`, and CA certificates as dependencies.

5. Keep `train.py` stable and avoid fake GPU signals.
   - Do not hardcode GPU or RTX 4090 messages.
   - Keep the existing `tf.config.list_physical_devices('GPU')` detection unless a minimal TensorFlow 2.21 compatibility adjustment is required.
   - Preserve model architecture, random seeds, optimizer, batch size, and training flow unless accuracy validation proves a real stack-level issue.

6. Update SavedModel export only if needed.
   - Test whether `conversion.py` still works under TensorFlow 2.21.
   - If `model.save(..., save_format='tf')` fails under newer Keras behavior, switch to a supported SavedModel export while preserving the `out/saved_model` directory and `out/saved_model.zip` artifact contract.

7. Build and iterate.
   - Run `docker build -t custom-ml-keras .` from the repo root.
   - Fix build blockers related to Ubuntu 24.04 package names, Python packaging, TensorFlow dependency resolution, or CUDA library discovery.

8. Validate GPU visibility.
   - Run `docker run --gpus all --rm custom-ml-keras python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"`.
   - Confirm TensorFlow sees at least one GPU and the runtime exposes NVIDIA device details.

9. Run the required training command.
   - Run `docker run --gpus all --rm -v $PWD:/app custom-ml-keras --data-directory /app/data --epochs 5 --learning-rate 0.01 --out-directory out/`.
   - Confirm `Training on: gpu`, real RTX 4090 runtime/device evidence, accuracy around 0.98, `Saving saved model OK`, and `out/saved_model.zip`.

10. Clean up.
    - Remove obsolete temporary Dockerfiles if any were created.
    - Do not commit changes.
