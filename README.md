# Custom Resnet50 example for Edge Impulse

This repository is an example on how to bring Resnet (built with Keras) into Edge Impulse. 

As a primer, read the [Bring your own model](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) page in the Edge Impulse docs.

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
3. Create a new Edge Impulse project, and add images (see [Adding sight to your sensors](https://docs.edgeimpulse.com/docs/tutorials/image-classification)).
4. Under **Create impulse**, set a square resolution (e.g. **224x224**), then add a 'Image' processing block (make sure it is set to **RGB**), and a Tansfer learning ML block.
5. Open a command prompt or terminal window.
6. Initialize the block:

    ```
    $ edge-impulse-blocks init
    # Answer the questions, select "Image classification" for 'What type of data does this model operate on?'
    ```

7. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

8. Build the container:

    ```
    $ docker build -t resnet .
    ```

9. Run the container to test the script (you don't need to rebuild the container if you make changes):

    ```
    $ docker run --rm -v $PWD:/app resnet --data-directory /app/data --epochs 10 --learning-rate 0.001 --out-directory out/
    ```

10. This creates two .tflite files and a saved model ZIP file in the 'out' directory.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

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

2. The block is now available under any of your projects via **Create impulse > Add learning block > Transfer learning (Images)**, then select the block via 'Choose a different model' on the 'Transfer learning' page.
