import os
import tensorflow as tf
import numpy as np
from akida_models import akidanet_imagenet
from keras import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import BatchNormalization, Conv2D, Softmax, ReLU
from cnn2snn import check_model_compatibility
from ei_tensorflow.constrained_object_detection import models, dataset, metrics, util

#WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', os.getcwd())
WEIGHTS_PREFIX = '/app'

def build_model(input_shape: tuple, alpha: float,
                num_classes: int, weight_regularizer=None) -> tf.keras.Model:
    """ Construct a constrained object detection model.

    Args:
        input_shape: Passed to AkidaNet construction.
        alpha: AkidaNet alpha value.
        num_classes: Number of classes, i.e. final dimension size, in output.

    Returns:
        Uncompiled keras model.

    Model takes (B, H, W, C) input and
    returns (B, H//8, W//8, num_classes) logits.
    """
    #! Create a quantized base model without top layers
    a_base_model = akidanet_imagenet(input_shape=input_shape,
                                     alpha=alpha,
                                     include_top=False,
                                     input_scaling=None)

    #! Get pretrained quantized weights and load them into the base model
    #! Available base models are:
    #! akidanet_imagenet_224.h5                      - float32 model, 224x224x3, alpha=1.00
    #! akidanet_imagenet_224_alpha_50.h5             - float32 model, 224x224x3, alpha=0.50
    #! akidanet_imagenet_224_alpha_25.h5             - float32 model, 224x224x3, alpha=0.25
    #! akidanet_imagenet_160.h5                      - float32 model, 160x160x3, alpha=1.00
    #! akidanet_imagenet_160_alpha_50.h5             - float32 model, 160x160x3, alpha=0.50
    #! akidanet_imagenet_160_alpha_25.h5             - float32 model, 160x160x3, alpha=0.25
    pretrained_weights = os.path.join(WEIGHTS_PREFIX , 'transfer-learning-weights/akidanet/akidanet_imagenet_224_alpha_50.h5')
    a_base_model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)
    a_base_model.trainable = True

    #! Default batch norm is configured for huge networks, let's speed it up
    # TODO: AkidaNet also requires it?
    for layer in a_base_model.layers:
        if type(layer) == BatchNormalization:
            layer.momentum = 0.9

    #! Cut AkidaNet where it hits 1/8th input resolution; i.e. (HW/8, HW/8, C)
    a_cut_point = a_base_model.get_layer('separable_5_relu')

    #! Now attach a small additional head on the AkidaNet
    a_model_part_head = Conv2D(filters=32, kernel_size=1, strides=1, padding='same',
                               kernel_regularizer=weight_regularizer)(a_cut_point.output)
    a_model_part = ReLU()(a_model_part_head)
    a_logits = Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='same',
                      activation=None, kernel_regularizer=weight_regularizer)(a_model_part)

    fomo_akida = Model(inputs=a_base_model.input, outputs=a_logits)

    #! Check if the model is sompatbile with Akida (fail quickly before training)
    compatible = check_model_compatibility(fomo_akida, input_is_image=True)
    if not compatible:
        print("Model is not compatible with Akida!")
        sys.exit(1)

    return fomo_akida

def train(num_classes: int, learning_rate: float, num_epochs: int,
            alpha: float, object_weight: int,
            train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset,
            best_model_path: str,
            input_shape: tuple,
            callbacks: 'list',
            quantize_function,
            qat_function,
            lr_finder: bool = False) -> tf.keras.Model:
    """ Construct and train a constrained object detection model.

    Args:
        num_classes: Number of classes in datasets. This does not include
            implied background class introduced by segmentation map dataset
            conversion.
        learning_rate: Learning rate for Adam.
        num_epochs: Number of epochs passed to model.fit
        alpha: Alpha used to construct AkidaNet. Pretrained weights will be
            used if there is a matching set.
        object_weight: The weighting to give the object in the loss function
            where background has an implied weight of 1.0.
        train_dataset: Training dataset of (x, (bbox, one_hot_y))
        validation_dataset: Validation dataset of (x, (bbox, one_hot_y))
        best_model_path: location to save best model path. note: weights
            will be restored from this path based on best val_f1 score.
        input_shape: The shape of the model's input
        lr_finder: TODO
    Returns:
        Trained keras model.

    Constructs a new constrained object detection model with num_classes+1
    outputs (denoting the classes with an implied background class of 0).
    Both training and validation datasets are adapted from
    (x, (bbox, one_hot_y)) to (x, segmentation_map). Model is trained with a
    custom weighted cross entropy function.
    """

    # nonlocal callbacks # type: ignore

    num_classes_with_background = num_classes + 1

    input_width_height = None
    width, height, input_num_channels = input_shape
    if width != height:
        raise Exception(f"Only square inputs are supported; not {input_shape}")
    input_width_height = width

    model = build_model(input_shape=input_shape,
                        alpha=alpha,
                        num_classes=num_classes_with_background,
                        weight_regularizer=tf.keras.regularizers.l2(4e-5))

    #! Derive output size from model
    model_output_shape = model.layers[-1].output.shape
    _batch, width, height, num_classes = model_output_shape
    if width != height:
        raise Exception(f"Only square outputs are supported; not {model_output_shape}")
    output_width_height = width

    #! Build weighted cross entropy loss specific to this model size
    weighted_xent = models.construct_weighted_xent_fn(model.output.shape, object_weight)

    #! Transform bounding box labels into segmentation maps
    def as_segmentation(ds):
        return ds.map(dataset.bbox_to_segmentation(output_width_height, num_classes_with_background)
                      ).batch(32, drop_remainder=False).prefetch(1)
    train_segmentation_dataset = as_segmentation(train_dataset)
    validation_segmentation_dataset = as_segmentation(validation_dataset)

    # Do an additional version of the validation dataset that is passed to the
    # centroid scoring callback ( which uses (x, (bb, labels)) ) _with_ the mapping
    validation_dataset_for_callback = validation_dataset.batch(32, drop_remainder=False).prefetch(1)

    #! Initialise bias of final classifier based on training data prior.
    util.set_classifier_biases_from_dataset(
        model, train_segmentation_dataset)

    if lr_finder:
        learning_rate = ei_tensorflow.lr_finder.find_lr(model, train_segmentation_dataset, weighted_xent)

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss=weighted_xent,
                    optimizer=opt)

    #! Create callback that will do centroid scoring on end of epoch against
    #! validation data. Include a callback to show % progress in slow cases.
    centroid_callback = metrics.CentroidScoring(validation_dataset_for_callback,
                                                output_width_height, num_classes_with_background)
    print_callback = metrics.PrintPercentageTrained(num_epochs)

    #! Include a callback for model checkpointing based on the best validation f1.
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(best_model_path,
            monitor='val_f1', save_best_only=True, mode='max',
            save_weights_only=True, verbose=0)

    model.fit(train_segmentation_dataset,
              validation_data=validation_segmentation_dataset,
              epochs=num_epochs,
              callbacks=callbacks + [centroid_callback, print_callback, checkpoint_callback],
              verbose=0)

    #! Restore best weights.
    model.load_weights(best_model_path)

    #! Add explicit softmax layer before export.
    softmax_layer = Softmax()(model.layers[-1].output)
    model = Model(model.input, softmax_layer)

    #! Check if model is compatible with Akida
    compatible = check_model_compatibility(model, input_is_image=True)
    if not compatible:
        print("Model is not compatible with Akida!")
        sys.exit(1)

    #! Quantize model to 4/4/8
    akida_model = quantize_function(keras_model=model)

    #! Perform quantization-aware training
    akida_model = qat_function(akida_model=akida_model,
                               train_dataset=train_segmentation_dataset,
                               validation_dataset=validation_segmentation_dataset,
                               optimizer=opt,
                               fine_tune_loss=weighted_xent,
                               fine_tune_metrics=None,
                               callbacks=callbacks + [centroid_callback, print_callback],
                               stopping_metric='val_f1',
                               fit_verbose=0)

    return model, akida_model
