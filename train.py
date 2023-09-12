import math, random
import numpy as np
import argparse, os, sys,logging
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

from keras import Model
from keras.layers import Activation, Dropout, Reshape, Flatten

from conversion import convert_to_tf_lite, save_saved_model

from akida_models.layer_blocks import dense_block
from akida_models import akidanet_imagenet

import cnn2snn

from brainchip.model import convert_akida_model
from brainchip.quantize import *
from brainchip.transfer import train

#WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', os.getcwd())
WEIGHTS_PREFIX = '/app'

# Load files
parser = argparse.ArgumentParser(description='Running custom Keras models in Edge Impulse')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set and convert into TF Dataset
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

MODEL_INPUT_SHAPE = X_train.shape[1:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

classes = Y_train.shape[1]
EPOCHS = args.epochs or 20
LEARNING_RATE = args.learning_rate or 0.0005

callbacks = []

BEST_MODEL_PATH = os.path.join(os.sep, 'tmp', 'best_model.hdf5')

# Available pretrained_weights are:
# akidanet_imagenet_224_alpha_100.h5            - float32 model, 224x224x3, alpha=1.00
# akidanet_imagenet_224_alpha_50.h5             - float32 model, 224x224x3, alpha=0.50
# akidanet_imagenet_224_alpha_25.h5             - float32 model, 224x224x3, alpha=0.25
# akidanet_imagenet_160_alpha_100.h5            - float32 model, 160x160x3, alpha=1.00
# akidanet_imagenet_160_alpha_50.h5             - float32 model, 160x160x3, alpha=0.50
# akidanet_imagenet_160_alpha_25.h5             - float32 model, 160x160x3, alpha=0.25
model, akida_model, akida_edge_model = train(train_dataset=train_dataset,
                                             validation_dataset=validation_dataset,
                                             num_classes=classes,
                                             pretrained_weights=os.path.join(WEIGHTS_PREFIX , 'transfer-learning-weights/akidanet/akidanet_imagenet_160_alpha_50.h5'),
                                             input_shape=MODEL_INPUT_SHAPE,
                                             learning_rate=LEARNING_RATE,
                                             epochs=EPOCHS,
                                             dense_layer_neurons=16,
                                             dropout=0.1,
                                             data_augmentation=False,
                                             callbacks=callbacks,
                                             alpha=0.5,
                                             best_model_path=BEST_MODEL_PATH,
                                             quantize_function=akida_quantize_model,
                                             qat_function=akida_perform_qat,
                                             edge_learning_function=None,
                                             additional_classes=None,
                                             neurons_per_class=None,
                                             X_train=X_train)

print('')
print('Training network OK')
print('')

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.
disable_per_channel_quantization = False

# Save the model to disk
save_saved_model(model, args.out_directory)

# Create tflite files (f32 / i8)
convert_to_tf_lite(model, args.out_directory, validation_dataset, MODEL_INPUT_SHAPE,
    'model.tflite', 'model_quantized_int8_io.tflite', disable_per_channel_quantization)

convert_akida_model(args.out_directory, akida_model,
                    'akida_model.fbz',
                    MODEL_INPUT_SHAPE)

print(os.listdir(args.out_directory))
