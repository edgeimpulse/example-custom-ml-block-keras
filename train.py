import argparse, os, sys, random, logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from conversion import save_saved_model

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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

classes = Y_train.shape[1]

MODEL_INPUT_SHAPE = X_train.shape[1:]
if len(MODEL_INPUT_SHAPE) != 3:
    raise ValueError(f'MobileNetV3 expects image input shaped (height, width, channels), got {MODEL_INPUT_SHAPE}')
if MODEL_INPUT_SHAPE[2] not in (1, 3):
    raise ValueError(f'MobileNetV3 expects 1-channel or 3-channel image input, got {MODEL_INPUT_SHAPE[2]} channels')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

# print GPU/CPU info
print('Training on:', 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu')
print('')

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []

# model architecture
inputs = layers.Input(shape=MODEL_INPUT_SHAPE)
x = inputs
if MODEL_INPUT_SHAPE[2] == 1:
    x = layers.Concatenate()([x, x, x])

backbone = tf.keras.applications.MobileNetV3Small(
    input_shape=x.shape[1:],
    include_top=False,
    weights='imagenet',
    pooling='avg',
    include_preprocessing=True,
    minimalistic=True,
)
backbone.trainable = False

x = backbone(x, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(classes, activation='softmax', name='y_pred')(x)
model = Model(inputs=inputs, outputs=outputs)

# this controls the learning rate
opt = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 32
train_dataset_batch = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset_batch = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], jit_compile=False)
model.fit(train_dataset_batch, epochs=args.epochs, validation_data=validation_dataset_batch, verbose=2, callbacks=callbacks)

print('')
print('Training network OK')
print('')

# Save the model to disk
save_saved_model(model, args.out_directory)
