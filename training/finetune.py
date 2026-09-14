import numpy as np
import tensorflow as tf

from training.train import compile_model, make_callbacks


def combine_replay_data(
    X_new,
    y_new,
    X_old=None,
    y_old=None,
    old_fraction=0.30,
    seed=42,
):
    if X_old is None or y_old is None or old_fraction <= 0:
        return X_new, y_new

    rng = np.random.default_rng(seed)

    new_count = len(X_new)
    old_count = int(new_count * old_fraction / max(1e-8, 1.0 - old_fraction))
    old_count = min(old_count, len(X_old))

    indices = rng.choice(len(X_old), size=old_count, replace=False)

    X_mix = np.concatenate([X_new, X_old[indices]], axis=0)
    y_mix = np.concatenate([y_new, y_old[indices]], axis=0)

    perm = rng.permutation(len(X_mix))
    return X_mix[perm], y_mix[perm]


def finetune_model(
    model_path,
    X_train,
    y_train,
    X_val,
    y_val,
    output_path,
    learning_rate=1e-5,
    epochs=10,
    batch_size=128,
):
    model = tf.keras.models.load_model(str(model_path), compile=False)
    compile_model(model, learning_rate=learning_rate)

    callbacks = make_callbacks(
        output_path,
        early_stopping_patience=4,
        reduce_lr_patience=2,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    return tf.keras.models.load_model(str(output_path), compile=False)
