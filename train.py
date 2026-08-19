import argparse, os, sys, random, logging, math
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


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got "{value}"')


def parse_last_layers(last_layers):
    parsed_layers = []
    if not last_layers or last_layers.strip() == '':
        return parsed_layers

    for option in last_layers.split(','):
        split = [part.strip() for part in option.split(':')]
        if len(split) != 2:
            raise ValueError(f'Failed to parse --last-layers, option: "{option}" cannot be parsed')

        name, value = split
        if name == 'dense':
            parsed_layers.append(layers.Dense(int(value), activation='relu'))
        elif name == 'dropout':
            parsed_layers.append(layers.Dropout(float(value)))
        else:
            raise ValueError(f'Failed to parse --last-layers, option: "{option}" key was not recognized (should be dense, dropout)')

    return parsed_layers


def parse_data_augmentation(data_augmentation):
    if not data_augmentation or data_augmentation.strip() == '':
        return []

    options = [option.strip().strip("'").strip('"') for option in data_augmentation.split(',')]
    valid_options = {'brightness', 'flip', 'crop'}
    for option in options:
        if option not in valid_options:
            raise ValueError(f'Failed to parse --data-augmentation, invalid value "{option}" (valid: brightness, flip, crop)')
    return options


# Load files
parser = argparse.ArgumentParser(description='Running custom Keras models in Edge Impulse')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--model-size', type=str, default='small', choices=['small', 'large'])
parser.add_argument('--use-pretrained-weights', type=parse_bool, nargs='?', const=True, default=True)
parser.add_argument('--freeze-percentage-of-layers', type=int, default=90)
parser.add_argument('--last-layers', type=str, default='dense: 32, dropout: 0.1')
parser.add_argument('--data-augmentation', type=str, default='')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--early-stopping', type=parse_bool, nargs='?', const=True, default=True)
parser.add_argument('--early-stopping-patience', type=int, default=5)
parser.add_argument('--early-stopping-min-delta', type=float, default=0.001)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()
augmentation_options = parse_data_augmentation(args.data_augmentation)
last_layers = parse_last_layers(args.last_layers)

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

if augmentation_options:
    def augment_image(image, label):
        if 'flip' in augmentation_options:
            image = tf.image.random_flip_left_right(image)

        if 'crop' in augmentation_options:
            resize_factor = tf.random.uniform([], 1.0, 1.2)
            new_height = tf.cast(tf.math.floor(resize_factor * MODEL_INPUT_SHAPE[0]), tf.int32)
            new_width = tf.cast(tf.math.floor(resize_factor * MODEL_INPUT_SHAPE[1]), tf.int32)
            image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
            image = tf.image.random_crop(image, size=MODEL_INPUT_SHAPE)

        if 'brightness' in augmentation_options:
            image = tf.image.random_brightness(image, max_delta=0.2)

        return image, label

    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

# print GPU/CPU info
print('Training on:', 'gpu' if len(tf.config.list_physical_devices('GPU')) > 0 else 'cpu')
print('')

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []
if args.early_stopping:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    ))

# model architecture
inputs = layers.Input(shape=MODEL_INPUT_SHAPE)
x = inputs
if MODEL_INPUT_SHAPE[2] == 1:
    x = layers.Concatenate()([x, x, x])

mobilenet_model = {
    'small': tf.keras.applications.MobileNetV3Small,
    'large': tf.keras.applications.MobileNetV3Large,
}[args.model_size]

backbone = mobilenet_model(
    input_shape=x.shape[1:],
    include_top=False,
    weights='imagenet' if args.use_pretrained_weights else None,
    pooling='avg',
    include_preprocessing=True,
    minimalistic=True,
)
if args.use_pretrained_weights:
    fine_tune_from = math.ceil(len(backbone.layers) * (args.freeze_percentage_of_layers / 100))
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_from]:
        layer.trainable = False

x = backbone(x, training=False)
for last_layer in last_layers:
    x = last_layer(x)
outputs = layers.Dense(classes, activation='softmax', name='y_pred')(x)
model = Model(inputs=inputs, outputs=outputs)

# this controls the learning rate
opt = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999)
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
train_dataset_batch = train_dataset.batch(args.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
validation_dataset_batch = validation_dataset.batch(args.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'], jit_compile=False)
model.fit(train_dataset_batch, epochs=args.epochs, validation_data=validation_dataset_batch, verbose=2, callbacks=callbacks)

print('')
print('Training network OK')
print('')

# Save the model to disk
save_saved_model(model, args.out_directory)
