#!/bin/bash
set -e

apt update && apt install -y ruby

# Patch cnn2snn library for Keras 2.11 compat
cd /usr/local/lib/python3.8/dist-packages/cnn2snn
FIND="from keras.utils.generic_utils import serialize_keras_object" REPLACE="from keras.saving.legacy.serialization import serialize_keras_object" \
    ruby -p -i -e "gsub(ENV['FIND'], ENV['REPLACE'])" quantization_layers.py
FIND="tf.keras.utils.deserialize_keras_object" REPLACE="tf.keras.saving.legacy.serialization.deserialize_keras_object" \
    ruby -p -i -e "gsub(ENV['FIND'], ENV['REPLACE'])" quantization_layers.py

cd /usr/local/lib/python3.8/dist-packages/akida_models
FIND="from keras.utils.generic_utils import custom_object_scope" REPLACE="from keras.utils import custom_object_scope" \
    ruby -p -i -e "gsub(ENV['FIND'], ENV['REPLACE'])" ./gamma_constraint.py
