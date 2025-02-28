# Custom Keras ML block example for Edge Impulse

This repository is an example on how to [add a custom learning block](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) to Edge Impulse. This repository contains a small fully-connected model built in Keras & TensorFlow. If you want to see a more complex example, see [edgeimpulse/efficientnet](https://github.com/edgeimpulse/example-custom-ml-block-efficientnet). Or if you're looking for the PyTorch version of this repository, see [edgeimpulse/example-custom-ml-block-pytorch](https://github.com/edgeimpulse/example-custom-ml-block-pytorch).

As a primer, read the [Custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) page in the Edge Impulse docs.

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
3. Create a new Edge Impulse project, and add data from the [continuous gestures](https://docs.edgeimpulse.com/docs/continuous-gestures) dataset.
4. Under **Create impulse** add a 'Spectral features' processing block, and a random ML block.
5. Open a command prompt or terminal window.
6. Initialize the block:

    ```
    $ edge-impulse-blocks init --clean
    ```

7. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

8. Build the container:

    ```
    $ docker build -t custom-ml-keras .
    ```

9. Run the container to test the script (you don't need to rebuild the container if you make changes):

    **macOS, Linux**

    ```
    $ docker run --rm -v $PWD:/app custom-ml-keras --data-directory /app/data --epochs 30 --learning-rate 0.01 --out-directory out/
    ```

    **Windows (Command prompt)**

    ```
    $ docker run --rm -v "%cd%":/app custom-ml-keras --data-directory /app/data --epochs 30 --learning-rate 0.01 --out-directory out/
    ```

    **Windows (Powershell)**

    ```
    $ docker run --rm -v ${PWD}$:/app custom-ml-keras --data-directory /app/data --epochs 30 --learning-rate 0.01 --out-directory out/
    ```

10. This creates a saved model ZIP file in the 'out' directory.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

#### Adding new arguments

To add new arguments, see [Custom learning blocks > Arguments to your script](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks#arguments-to-your-script).

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects via **Create impulse > Add new learning block**.

## Changing the block type (e.g. image classification, object detection or regression)

If you want to change the block type because you're classifying a different data type, or build a model with a different output format, run:

```
$ rm parameters.json  .ei-block-config
$ edge-impulse-blocks init
```

And answer the wizard. This'll create a new parameters.json file.
