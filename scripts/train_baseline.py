import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import (
    load_train_val_test,
    make_sample_weights,
)
from models.baseline_model import (
    BASELINE_ARCHITECTURE,
    build_baseline_model,
)
from nas.analytical_cost import estimate_architecture_cost
from nas.cost import estimate_macs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/baseline",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    (
        X_train, y_train, y_train_int, _,
        X_val, y_val, y_val_int, _,
        X_test, y_test, y_test_int, _,
    ) = load_train_val_test(
        augment_train=True,
        seed=args.seed,
    )

    del X_test, y_test, y_test_int

    sample_weight = make_sample_weights(
        y_train_int
    )

    model = build_baseline_model()
    cost = estimate_architecture_cost(
        BASELINE_ARCHITECTURE
    )

    if model.count_params() != cost.params:
        raise RuntimeError("Baseline Params mismatch.")
    if estimate_macs(model) != cost.macs:
        raise RuntimeError("Baseline MACs mismatch.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.01
        ),
        metrics=["accuracy"],
    )

    checkpoint = out / "best_baseline.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        sample_weight=sample_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    best = tf.keras.models.load_model(
        checkpoint,
        compile=False,
    )

    probs = best.predict(
        X_val,
        batch_size=max(256, args.batch_size),
        verbose=0,
    )
    pred = np.argmax(probs, axis=1)

    metrics = {
        "architecture": BASELINE_ARCHITECTURE,
        "validation_samples": int(len(y_val_int)),
        "accuracy": float(
            accuracy_score(y_val_int, pred)
        ),
        "macro_f1": float(
            f1_score(
                y_val_int,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_val_int,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_val_int,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "params": int(cost.params),
        "macs": int(cost.macs),
        "memory_mb": float(cost.memory_mb),
        "test_set_used_for_selection": False,
    }

    path = out / "baseline_metrics.json"
    path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
