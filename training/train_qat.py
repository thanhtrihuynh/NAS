import tensorflow as tf


def compile_qat_model(
    model,
    learning_rate=1e-5
):
    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=
                learning_rate
        )
    )

    loss = (
        tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.03
        )
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    return model


def make_qat_callbacks(
    checkpoint_path
):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


def train_qat_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    checkpoint_path,
    epochs=15,
    batch_size=128,
    learning_rate=1e-5
):
    compile_qat_model(
        model,
        learning_rate=
            learning_rate
    )

    callbacks = (
        make_qat_callbacks(
            checkpoint_path
        )
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    return history