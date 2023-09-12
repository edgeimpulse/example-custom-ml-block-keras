import math, random
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

from keras import Model
from keras.layers import Activation, Dropout, Reshape, Flatten

from akida_models.layer_blocks import dense_block
from akida_models import akidanet_imagenet

import cnn2snn

BATCH_SIZE = 32

#! Implements the data augmentation policy
def augmentation_function(input_shape: tuple):
    def augment_image(image, label):
        # Flips the image randomly
        image = tf.image.random_flip_left_right(image)

        #! Increase the image size, then randomly crop it down to
        #! the original dimensions
        resize_factor = random.uniform(1, 1.2)
        new_height = math.floor(resize_factor * input_shape[0])
        new_width = math.floor(resize_factor * input_shape[1])
        image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
        image = tf.image.random_crop(image, size=input_shape)

        #! Vary the brightness of the image
        image = tf.image.random_brightness(image, max_delta=0.2)

        return image, label

    return augment_image

def train(train_dataset: tf.data.Dataset,
          validation_dataset: tf.data.Dataset,
          num_classes: int,
          pretrained_weights: str,
          input_shape: tuple,
          learning_rate: int,
          epochs: int,
          dense_layer_neurons: int,
          dropout: float,
          data_augmentation: bool,
          callbacks,
          alpha: float,
          best_model_path: str,
          quantize_function,
          qat_function,
          edge_learning_function=None,
          additional_classes=None,
          neurons_per_class=None,
          X_train=None):
    #! Create a quantized base model without top layers
    base_model = akidanet_imagenet(input_shape=input_shape,
                                classes=num_classes,
                                alpha=alpha,
                                include_top=False,
                                input_scaling=None,
                                pooling='avg')

    base_model.load_weights(pretrained_weights, by_name=True, skip_mismatch=True)

    #! Freeze that base model, so it won't be trained
    base_model.trainable = False

    output_model = base_model.output
    output_model = Flatten()(output_model)
    if dense_layer_neurons > 0:
        output_model = dense_block(output_model,
                                units=dense_layer_neurons,
                                add_batchnorm=False,
                                add_activation=True)
    if dropout > 0:
        output_model = Dropout(dropout)(output_model)
    output_model = dense_block(output_model,
                            units=num_classes,
                            add_batchnorm=False,
                            add_activation=False)
    output_model = Activation('softmax')(output_model)
    output_model = Reshape((num_classes,))(output_model)

    #! Build the model
    model = Model(base_model.input, output_model)

    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if data_augmentation:
        train_dataset = train_dataset.map(augmentation_function(input_shape),
                                          num_parallel_calls=tf.data.AUTOTUNE)

    #! This controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
    validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

    #! Train the neural network
    model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=2, callbacks=callbacks)

    print('')
    print('Initial training done.', flush=True)
    print('')

    #! Unfreeze the model before QAT
    model.trainable = True

    #! Quantize model to 4/4/8
    akida_model = quantize_function(keras_model=model)

    #! Do a quantization-aware training
    akida_model = qat_function(akida_model=akida_model,
                               train_dataset=train_dataset,
                               validation_dataset=validation_dataset,
                               optimizer=opt,
                               fine_tune_loss='categorical_crossentropy',
                               fine_tune_metrics=['accuracy'],
                               callbacks=callbacks)
    #! Optionally, build the edge learning model
    if edge_learning_function:
        akida_edge_model = edge_learning_function(quantized_model=akida_model,
                                                  X_train=X_train,
                                                  train_dataset=train_dataset,
                                                  validation_dataset=validation_dataset,
                                                  callbacks=callbacks,
                                                  optimizer=opt,
                                                  fine_tune_loss='categorical_crossentropy',
                                                  fine_tune_metrics=['accuracy'],
                                                  additional_classes=additional_classes,
                                                  neurons_per_class=neurons_per_class,
                                                  num_classes=num_classes,
                                                  qat_function=qat_function)
    else:
        akida_edge_model = None


    return model, akida_model, akida_edge_model
