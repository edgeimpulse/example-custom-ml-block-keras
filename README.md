# Custom EfficientNet example for Edge Impulse

This repository is an example on how to bring EfficientNet (built using Keras) into Edge Impulse.

As a primer, read the [Custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) page in the Edge Impulse docs.

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
3. Create a new Edge Impulse project, and add an image dataset (see [Adding sight to your sensors](https://docs.edgeimpulse.com/docs/tutorials/image-classification)).
4. Under **Create impulse**, set the resolution to a square size (e.g. **224x224**), then add a 'Image' processing block (make sure it's set to RGB), and a 'Transfer Learning (Images)' ML block.
5. Open a command prompt or terminal window.
6. Initialize the block:

    ```
    $ edge-impulse-blocks init
    ```

7. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

8. Build the container:

    ```
    $ docker build -t efficientnet .
    ```

9. Run the container to test the script (you don't need to rebuild the container if you make changes):

    ```
    $ docker run --rm -v $PWD:/app efficientnet --data-directory /app/data --epochs 20 --learning-rate 0.001 --out-directory out/
    ```

10. This creates two .tflite files and a saved model ZIP file in the 'out' directory.

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

2. The block is now available under any of your projects via **Create impulse > Add learning block**, then select the block via 'Choose a different model' on the 'Transfer learning' page.

## Changing the block type (e.g. image classification, object detection or regression)

If you want to change the block type because you're classifying a different data type, or build a model with a different output format, run:

```
$ rm parameters.json  .ei-block-config
$ edge-impulse-blocks init
```

And answer the wizard. This'll create a new parameters.json file.
