import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam
from conversion import convert_to_tf_lite, save_saved_model

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Sync HRV Logger files to Edge Impulse')
parser.add_argument('--x-file', type=str, required=False)
parser.add_argument('--y-file', type=str, required=False)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--validation-set-size', type=float, required=True)
parser.add_argument('--input-shape', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=False)

args = parser.parse_args()

# for --x-file, --y-file, --out-directory use the defaults (used by Edge Impulse), if not passed in
x_file = args.x_file if args.x_file else '/home/X_train_features.npy'
y_file = args.y_file if args.y_file else '/home/y_train.npy'
out_directory = args.out_directory if args.out_directory else '/home'

if not os.path.exists(x_file):
    print('--x-file argument', x_file, 'does not exist', flush=True)
    exit(1)
if not os.path.exists(y_file):
    print('--y-file argument', y_file, 'does not exist', flush=True)
    exit(1)
if not os.path.exists(out_directory):
    os.mkdir(out_directory)

X = np.load(x_file)
Y = np.load(y_file)[:,0]

classes = np.max(Y)

# get the shape of the input, and reshape the features
MODEL_INPUT_SHAPE = tuple([ int(x) for x in args.input_shape.replace('(', '').replace(')', '').split(',') ])
X = X.reshape(tuple([ X.shape[0] ]) + MODEL_INPUT_SHAPE)

# convert Y to a categorical vector
Y = tf.keras.utils.to_categorical(Y - 1, classes)

# split in train/validate set and convert into TF Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.validation_set_size, random_state=1)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []

# model architecture
model = Sequential()
model.add(Dense(20, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(10, activation='relu',
    activity_regularizer=tf.keras.regularizers.l1(0.00001)))
model.add(Dense(classes, activation='softmax', name='y_pred'))

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
save_saved_model(model, out_directory)

# Create tflite files (f32 / i8)
convert_to_tf_lite(model, out_directory, validation_dataset, MODEL_INPUT_SHAPE,
    'model.tflite', 'model_quantized_int8_io.tflite', disable_per_channel_quantization)
