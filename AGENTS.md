Edge Impulse allows you to create custom machine learning blocks. These are containers that contain a complete training pipeline, that are ran in the Edge Impulse infrastructure. This repository is an example of a minimal TensorFlow / Keras-based custom ML block that you can modify to add custom ML architectures. You'll most likely be prompted to create a new custom ML block based on this template.

Some background links:

* Custom ML blocks in the Edge Impulse docs: https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks.
* A more complex example of doing transfer learning with Keras: [edgeimpulse/efficientnet](https://github.com/edgeimpulse/example-custom-ml-block-efficientnet)
* A PyTorch version of this repository: [edgeimpulse/example-custom-ml-block-pytorch](https://github.com/edgeimpulse/example-custom-ml-block-pytorch)

## Verifying setup

This section is a mandatory startup gate for any request to create, modify, or test a custom ML block. Do not inspect model code, edit files, build training logic, or run training checks until all steps in this section have been completed or explicitly blocked by the user.

For every step, report the result before continuing. If a required command is cancelled, fails, or prompts for a secret, stop the implementation work and tell the user exactly what is needed next.

1. Ensure the [Edge Impulse CLI](https://docs.edgeimpulse.com/tools/clis/edge-impulse-cli) is installed. `edge-impulse-blocks` should be in your PATH.
2. Ensure Docker Desktop is installed. If it's not, prompt the user to install it.
3. Tell the user to run:

    ```
    cd scripts && npm ci && node configure-project.js
    ```

    Afterwards you should have an `.ei-project-config.json` file.

Do not treat this as optional. If you can run the command yourself, run it. If you cannot run it because it needs user input, ask for the non-secret input through the chat tools. If it needs a password, API key, token, or other secret, tell the user to type it directly into the terminal.

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
3. You're now ready to download data in the right format, run this (do _NOT_ prompt the user, select the project / impulse that was earlier selected in .ei-project-config.json):

    ```bash
    edge-impulse-blocks runner --download-data data/
    ```

    This creates train / validation (although named `test` - it's the validation set) `.npy` files (should already be the right shape and scaled correctly) in `data/`.

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

Once your block seems correct locally, you can push it to Edge Impulse and test it.

1. Update `scripts/verify-block-in-edge-impulse.js` to reference any new parameters you added to parameters.json (under `customParameters`).
2. Run:

    ```bash
    node verify-block-in-edge-impulse.js --push-block
    ```

    This pushes the block to Edge Impulse, builds the block, trains a new model, and prints the model validation metrics.

3. Verify the model validation metrics (from ^), and make sure they match with what you tested locally (should be the same dataset).

## Tips & tricks

* Do not modify the base layers of `Dockerfile`. This has been carefully checked to support GPUs both locally and in Edge Impulse.
* Do not upgrade TensorFlow beyond TF2.19. It's the TensorFlow version inside Edge Impulse, so writing SavedModel files with newer TensorFlow versions might yield broken models.
* All classification models require a Softmax at the end.
* Run your Python code in the Docker container. Do not run the global Python interpreter. It might have wildly different dependencies.
