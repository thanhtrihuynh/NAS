import tensorflow as tf

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE


def compile_model(model, learning_rate=LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
        metrics=["accuracy"],
    )
    return model


def make_callbacks(
    save_path,
    early_stopping_patience=7,
    reduce_lr_patience=3,
    min_delta=1e-4,
):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(save_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=min_delta,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            mode="min",
            verbose=1,
        ),
    ]


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    save_path,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    early_stopping_patience=7,
):
    callbacks = make_callbacks(
        save_path,
        early_stopping_patience=early_stopping_patience,
    )

    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
