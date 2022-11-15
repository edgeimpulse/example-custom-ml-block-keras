import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D,Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization,TimeDistributed, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from conversion import convert_to_tf_lite, save_saved_model
from tensorflow import keras
from tensorflow.keras import layers

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Restnet 50 models in Edge Impulse')
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

classes = Y_train.shape[1]

MODEL_INPUT_SHAPE = X_train.shape[1:]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# Weights file
dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(dir_path, 'transfer-learning-weights', 'keras','resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []

# Resnet50 requires data in BGR format, and specifically normalized
# but at Edge Impulse we deal with image data in RGB format scaled 0..1
# so here we'll do some in-graph magic to normalize

# first we create a matrix for which we can subtract input data (after RGB=>BGR and scaling)
# these values come from https://github.com/keras-team/keras/blob/506e36a6c967f62342aa49e2828b28edd2a59bc8/keras/applications/imagenet_utils.py#L218
scale_matrix = np.ones((1,) + MODEL_INPUT_SHAPE)
scale_matrix[:,:,0] = 103.939
scale_matrix[:,:,1] = 116.779
scale_matrix[:,:,2] = 123.68

# base model architecture (transfer learning, no trainable layers)
base_model = tf.keras.applications.ResNet50(include_top=False, weights=weights_path, pooling='avg', classes=classes)
base_model.trainable = False

input = keras.Input(shape=(MODEL_INPUT_SHAPE))
x = input
# RGB => BGR
x = layers.Lambda(function=lambda x: x[..., -1::-1])(x)
# 0..1 => 0..255
x = layers.Rescaling(scale=255)(x)
# normalize (see scale_matriux above)
x = layers.Subtract()([ x, scale_matrix ])
# ResNet50 (frozen layers, see above)
x = base_model(x)
# Dense layer
x = layers.Dense(16, activation='relu')(x)
# Some dropout and a flatten layer
x = layers.Dropout(0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(classes, activation='softmax')(x)
output = x

model = keras.Model(inputs=input, outputs=output)

# this controls the learning rate
opt = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 32
train_dataset_batch = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset_batch = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset_batch, epochs=args.epochs, validation_data=validation_dataset_batch, verbose=2, callbacks=callbacks)

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
