import tensorflow as tf

def build_edge_learning_model(quantized_model,
                              X_train,
                              train_dataset: tf.data.Dataset,
                              validation_dataset: tf.data.Dataset,
                              callbacks,
                              optimizer: str,
                              fine_tune_loss: str,
                              fine_tune_metrics: 'list[str]',
                              additional_classes: int,
                              neurons_per_class: int,
                              num_classes: int,
                              qat_function = None):
    from ei_tensorflow.brainchip.model import get_akida_converted_model
    import cnn2snn
    import akida
    import numpy as np
    from math import ceil
    import sys


    print("Looking for the feature extractor")
    feature_extractor_type =  cnn2snn.quantization_layers.QuantizedReLU
    feature_extractor_found = False
    for layer in reversed(quantized_model.layers):
        if isinstance(layer, feature_extractor_type):
            print("")
            print(f"The assumed feature extractor layer is: {layer.name}")
            print("")
            print("Setting feature extractor bitwidth to 1")
            quantized_model = cnn2snn.quantize_layer(quantized_model, layer, bitwidth=1)
            feature_extractor_found = True
            break

    if not feature_extractor_found:
        print("EI_LOG_LEVEL=error ERROR: Can't find the feature extractor! Edge Learning model can't be built.")
        print("EI_LOG_LEVEL=info Try to modify 'feature_extractor_type' in the Keras Expert Mode")
        sys.exit(1)

    print("Looking for the feature extractor OK")
    #! After quantizing feature extractor layer to 1 bit, we need to retrain the model to recover the accuracy
    if qat_function:
        print("")
        print("Performing quantization-aware training...")
        quantized_model = qat_function(akida_model=quantized_model,
                                       train_dataset=train_dataset,
                                       validation_dataset=validation_dataset,
                                       optimizer=optimizer,
                                       fine_tune_loss=fine_tune_loss,
                                       fine_tune_metrics=fine_tune_metrics,
                                       callbacks=callbacks)
        print("Performing quantization-aware training OK")
        print("")
    else:
        print("EI_LOG_LEVEL=warn WARNING: QAT function not defined! Quantized model won't be retrained!")

    akida_edge_model = get_akida_converted_model(quantized_model, MODEL_INPUT_SHAPE)

    #! Build edge learning compatible model
    # TODO: slice model after feature extractor?
    akida_edge_model.pop_layer()
    layer_fc = akida.FullyConnected(name='akida_edge_layer',
                                    units=additional_classes * neurons_per_class,
                                    activation=False) 
    akida_edge_model.add(layer_fc)
    print('Building edge compatible model OK')

    print('Compiling edge learning model')
    #! Calculate suggested number of weights as described in:
    #! https://doc.brainchipinc.com/examples/edge/plot_1_edge_learning_kws.html#prepare-akida-model-for-learning
    num_samples = ceil(0.1 * X_train.shape[0])
    sparsities = akida.evaluate_sparsity(akida_edge_model, np.array(X_train[:num_samples]*255, dtype=np.uint8))
    output_density = 1 - sparsities[akida_edge_model.layers[-2]]
    avg_spikes = akida_edge_model.layers[-2].output_dims[-1] * output_density

    #! Fix the number of weights to 1.2 times the average number of output spikes
    num_weights = int(1.2 * avg_spikes)
    print("===========================================")
    print(f"The number of weights is: {num_weights}")
    print("===========================================")

    try:
        akida_edge_model.compile(num_weights=num_weights,
                                num_classes=additional_classes,
                                learning_competition=0.1)
    except ValueError as err:
        print(f"EI_LOG_LEVEL=error ERROR: Can't compile Edge Learning model: {err}")
        print(f"EI_LOG_LEVEL=error ERROR: If the estimated number of weights is 0, try to increase the number of training cycles")
        sys.exit(1)
    print('Compiling edge learning model OK')
    print('')        

    return akida_edge_model

