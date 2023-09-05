import tensorflow as tf


def akida_quantize_model(
    keras_model,
    weight_quantization: int = 4,
    activ_quantization: int = 4,
    input_weight_quantization: int = 8,
):
    import cnn2snn

    print("Performing post-training quantization...")
    akida_model = cnn2snn.quantize(
        keras_model,
        weight_quantization=weight_quantization,
        activ_quantization=activ_quantization,
        input_weight_quantization=input_weight_quantization,
    )
    print("Performing post-training quantization OK")
    print("")

    return akida_model


def akida_perform_qat(
    akida_model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    optimizer: str,
    fine_tune_loss: str,
    fine_tune_metrics: "list[str]",
    callbacks,
    stopping_metric: str = "val_accuracy",
    fit_verbose: int = 2,
    qat_epochs: int = 30,
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=stopping_metric,
        mode="max",
        verbose=1,
        min_delta=0,
        patience=10,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    print("Running quantization-aware training...")
    akida_model.compile(
        optimizer=optimizer, loss=fine_tune_loss, metrics=fine_tune_metrics
    )

    akida_model.fit(
        train_dataset,
        epochs=qat_epochs,
        verbose=fit_verbose,
        validation_data=validation_dataset,
        callbacks=callbacks,
    )

    print("Running quantization-aware training OK")
    print("")

    return akida_model
