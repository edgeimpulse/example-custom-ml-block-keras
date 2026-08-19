# Simple Ubuntu 24.04 base block with CUDA setup already
FROM public.ecr.aws/z9b3d4t5/ei-custom-ml-block-base:v1.95.5-test-9e8dfa82

# https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# Ensure we can output a valid Keras SavedModel (not a TF one) - so the data explorer works in Studio
ENV TF_USE_LEGACY_KERAS=1

# Copy Python requirements in and install them (--break-system-packages is required if we don't use a venv)
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages -r requirements.txt

# Copy the rest of your training scripts in
COPY . ./

# And tell us where to run the pipeline
ENTRYPOINT ["python3", "-u", "train.py"]
