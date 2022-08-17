# Custom Keras ML block example for Edge Impulse

This repository is an example on how to bring a [custom transfer learning model](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) into Edge Impulse. This repository contains a small fully-connected model built in Keras & TensorFlow. If you want to see a more complex PyTorch example, see [edgeimpulse/yolov5](https://github.com/edgeimpulse/yolov5).  Or if you're looking for the PyTorch version of this repository, see [edgeimpulse/example-custom-ml-block-keras](https://github.com/edgeimpulse/example-custom-ml-block-pytorch).

As a primer, read the [Adding custom transfer learning models](https://docs.edgeimpulse.com/docs/adding-custom-transfer-learning-models) page in the Edge Impulse docs.

To test this locally:

1. Create a new Edge Impulse project, and add data from the [continuous gestures](https://docs.edgeimpulse.com/docs/continuous-gestures) dataset.
1. Under **Create impulse** add a 'Spectral features' processing block, and a random ML block.
1. Generate features for the DSP block.
1. Then go to **Dashboard** and download the 'Spectral features training data' and 'Spectral features training labels' files.
1. Create a new folder in this repository named `home` and copy the downloaded files in under the names: `X_train_features.npy` and `y_train.npy`.
1. Build the container:

    ```
    $ docker build -t custom-ml .
    ```

1. Run the container to test:

    ```
    $ docker run --rm -v $PWD/home:/home custom-ml --epochs 1 --learning-rate 0.01 --validation-set-size 0.2 --input-shape "(33)"
    ```

1. This should have created two .tflite files in the 'home' directory.

Now you can initialize the block to Edge Impulse:

```
$ edge-impulse-blocks init
# Answer the questions, select "other" for 'What type of data does this model operate on?'
```

And push the block:

```
$ edge-impulse-blocks push
```

The block is now available under any project that's owned by your organization.
