Edge Impulse allows you to create custom machine learning blocks. These are containers that contain a complete training pipeline, that are ran in the Edge Impulse infrastructure. This repository is an example of a minimal TensorFlow / Keras-based custom ML block that you can modify to add custom ML architectures. You'll most likely be prompted to create a new custom ML block based on this template.

Some background links:

* Custom ML blocks in the Edge Impulse docs: https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks.
* A more complex example of doing transfer learning with Keras: [edgeimpulse/efficientnet](https://github.com/edgeimpulse/example-custom-ml-block-efficientnet)
* A PyTorch version of this repository: [edgeimpulse/example-custom-ml-block-pytorch](https://github.com/edgeimpulse/example-custom-ml-block-pytorch)

## Verifying setup

Before you start:

1. Ensure the [Edge Impulse CLI](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli) is installed. `edge-impulse-blocks` should be in your PATH.
2. Ensure Docker Desktop is installed. If it's not, prompt the user to install it.
3. Prompt the user for an API key to an Edge Impulse project. This project should match the type of the custom ML block they want to develop (e.g. image classification, object detection, or plain classification/regression). Verify that you can make a `GET` request to `https://studio.edgeimpulse.com/v1/api/projects/api-key-info` (set `x-api-key` header to the API key) - and that the `role` (in the response of the GET request) of the API key is `admin`.
4. Prompt the user for an impulse (`GET` request to `https://studio.edgeimpulse.com/v1/api/PROJECT_ID/impulses`) (use PROJECT_ID from step 3).
5. Create a file `.ei-info` with:

    ```
    {
        "projectId": XXX,
        "projectApiKey": YYY,
        "impulseId": ZZZ
    }
    ```

    (With the values retrieved in the previous steps)

## Preparing a new block

We'll need some metadata and configuration for the block. Re-run this every time the user asks you to start a new architecture:

1. If an `.ei-block-config` file does not exist:
    * Run `edge-impulse-blocks init`. If this prompts for a login -> ask the user. For other questions: if you know the answer -> answer it; otherwise relay question to the user.
2. Update `info` properties in `parameters.json`.
    * For `operatesOn` set this to:
        * `other` - Non-image and non-audio Classification.
        * `audio` - Audio classification.
        * `image` - Image classification.
        * `regression` - Any regression.
        * `object_detection` - Object detection.
        * `anomaly_detection` - Non-image anomaly detection.
        * `visual_anomaly_detection` - Image anomaly detection.
    * If `operatesOn` is `image`, `object_detection` or `visual_anomaly_detection` you also need to set `imageInputScaling` to one of: `0..1 | -1..1 | -128..127 | 0..255 | torch | bgr-subtract-imagenet-mean`. This is how image data will be preprocessing (e.g. scaled 0..255 or 0..1) before passing it to your network. Pick whatever is most suitable for the model architecture, e.g. make sure it matches the any transfer learning base model. If you don't care, or if you're building an architecture from scratch -> prefer `0..1`.
    * If `operatesOn` is `object_detection`, set `objectDetectionLastLayer` to one of: `mobilenet-ssd | fomo | yolov2-akida | yolov5 | yolov5v5-drpai | yolox | yolov7 | tao-retinanet | tao-ssd | tao-yolov3 | tao-yolov4`. See https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-learning-blocks#object-detection-output-layers for more information.
3. You're now ready to download data in the right format, run:

    ```bash
    edge-impulse-blocks runner --download-data data/

    # If this prompts for a project -> select the same project that matches .ei-api-key
    # If this prompts for an impulse -> ask the user. Also store the impulse that the user picked in .
    ```

    This creates train / validation (although named `test` - it's the validation set) `.npy` files (should already be the right shape and scaled correctly) in `data/`.

4.


## Training a model

Prefer to run in Docker (although you could create a venv if you want to test some stuff out quickly):

1. Build the container:

    ```bash
    docker build -t custom-ml-keras .
    ```

2. Train a model:

    ```bash
    docker run --network=none --rm -v $PWD:/app custom-ml-keras --data-directory /app/data --epochs 30 --learning-rate 0.01 --out-directory out/
    ```

    > `--network=none` is set here because the ML block will not have internet access when pushed to Edge Impulse! Add any files etc. that you need to the Docker container.

You're now ready to implement the new architecture. You can modify `train.py` (and other Python) files without having to rebuild the container. Once training finishes you should have a saved_model.zip file in the `out/` directory.

## Parameters

If you have new parameters you want to add to the block, add them as arguments to your training scripts (see `argparse`); and then also add them to the `parameters` section `parameters.json`. The latter will ensure there's UI rendered to configure the new parameters. See the `DSPParameterItem` spec in https://docs.edgeimpulse.com/tools/specifications/files/parameters-json for all options.

## Pushing the architecture to Edge Impulse and testing it

Once your block seems correct locally, you can push it to Edge Impulse and test it. To push your block, run:

```bash
edge-impulse-blocks push
```

## Tips & tricks

* Do not modify the base layers of `Dockerfile`. This has been carefully checked to support GPUs both locally and in Edge Impulse.
* Do not upgrade TensorFlow beyond TF2.19. It's the TensorFlow version inside Edge Impulse, so writing SavedModel files with newer TensorFlow versions might yield broken models.
* All classification models require a Softmax at the end.
