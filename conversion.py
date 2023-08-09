import io, os, shutil
import tensorflow as tf
import numpy as np

def get_concrete_function(keras_model, input_shape):
    # To produce an optimized model, the converter needs to see a static batch dimension.
    # At this point our model has an unspecified batch dimension, so we need to set it to 1.
    # See: https://github.com/tensorflow/tensorflow/issues/42286#issuecomment-681183961
    input_shape_with_batch = (1,) + input_shape
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(input_shape_with_batch, keras_model.inputs[0].dtype))
    return concrete_func

def run_converter(converter: tf.lite.TFLiteConverter):
    # The converter outputs some garbage that we don't want to end up in the user's log,
    # so we have to catch the c stdout/stderr and filter the things we don't want to keep.
    # TODO: Wrap this up more elegantly in a single 'with'
    converted_model = converter.convert()
    return converted_model

def convert_float32(concrete_func, keras_model, dir_path, filename):
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        # Restrict the supported types to avoid ops that are not TFLM compatible
        converter.target_spec.supported_types = [
            tf.dtypes.float32,
            tf.dtypes.int8
        ]
        tflite_model = run_converter(converter)
        open(os.path.join(dir_path, filename), 'wb').write(tflite_model)
        return tflite_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite float32 model:')
        print(err)


# Declare a generator that can feed the TensorFlow Lite converter during quantization
def representative_dataset_generator(validation_dataset):
    def gen():
        for data, _ in validation_dataset.take(-1).as_numpy_iterator():
            yield [tf.convert_to_tensor([data])]
    return gen

def convert_int8_io_int8(concrete_func, keras_model, dataset_generator,
                         dir_path, filename, disable_per_channel = False):
    try:
        converter_quantize = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
        if disable_per_channel:
            converter_quantize._experimental_disable_per_channel = disable_per_channel
            print('Note: Per channel quantization has been automatically disabled for this model. '
                  'You can configure this in Keras (expert) mode.')
        converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_quantize.representative_dataset = dataset_generator
        # Force the input and output to be int8
        converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Restrict the supported types to avoid ops that are not TFLM compatible
        converter_quantize.target_spec.supported_types = [tf.dtypes.int8]
        converter_quantize.inference_input_type = tf.int8
        converter_quantize.inference_output_type = tf.int8
        tflite_quant_model = run_converter(converter_quantize)
        open(os.path.join(dir_path, filename), 'wb').write(tflite_quant_model)
        return tflite_quant_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite int8 quantized model:')
        print(err)

def convert_to_tf_lite(model, dir_path,
                      validation_dataset, model_input_shape, model_filenames_float,
                      model_filenames_quantised_int8, disable_per_channel = False):

    dataset_generator = representative_dataset_generator(validation_dataset)
    concrete_func = get_concrete_function(model, model_input_shape)

    print('Converting TensorFlow Lite float32 model...')
    tflite_model = convert_float32(concrete_func, model, dir_path, model_filenames_float)
    print('Converting TensorFlow Lite float32 model OK')
    print('')

    print('Converting TensorFlow Lite int8 model...')
    tflite_quant_model = convert_int8_io_int8(concrete_func, model, dataset_generator,
                                              dir_path, model_filenames_quantised_int8,
                                              disable_per_channel)
    print('Converting TensorFlow Lite int8 model OK')
    print('')

    return model, tflite_model, tflite_quant_model

def save_saved_model(model, out_directory):
    print('Saving saved model...')

    saved_model_path = os.path.join(out_directory, 'saved_model')
    model.save(saved_model_path, save_format='tf')
    shutil.make_archive(saved_model_path,
                        'zip',
                        root_dir=os.path.dirname(saved_model_path),
                        base_dir='saved_model')

    print('Saving saved model OK')
    print('')
