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
from models.nas_model import build_nas_model


PRIORITY = {
    "DOMINATES_BASELINE": 1,
    "HIGHER_ACCURACY_TRADEOFF": 2,
    "NEAR_BASELINE_EFFICIENCY": 3,
    "AGGRESSIVE_EFFICIENCY_TRADEOFF": 4,
    "OTHER_TRADEOFF": 5,
    "DOMINATED_BY_BASELINE": 6,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = json.loads(
        Path(args.candidates).read_text(
            encoding="utf-8"
        )
    )

    rows.sort(
        key=lambda x: (
            PRIORITY.get(
                x.get(
                    "baseline_category",
                    "OTHER_TRADEOFF",
                ),
                5,
            ),
            -float(x["macro_f1"]),
            int(x["macs"]),
            int(x["params"]),
        )
    )

    selected = rows[:args.top_k]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    results = []

    for rank, c in enumerate(selected, start=1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(
            args.seed + rank
        )

        model = build_nas_model(
            c["architecture"]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=3e-4
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.01
            ),
            metrics=["accuracy"],
        )

        sub = out / (
            f"rank_{rank:02d}"
            f"_seed_{c.get('_source_seed','x')}"
            f"_id_{c.get('id','x')}"
        )
        sub.mkdir(parents=True, exist_ok=True)

        checkpoint = sub / "best.keras"

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

        result = {
            "rank": rank,
            "source_seed": c.get("_source_seed"),
            "source_candidate_id": c.get("id"),
            "architecture": c["architecture"],
            "short_macro_f1": float(c["macro_f1"]),
            "full_val_accuracy": float(
                accuracy_score(y_val_int, pred)
            ),
            "full_val_macro_f1": float(
                f1_score(
                    y_val_int,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "full_val_macro_precision": float(
                precision_score(
                    y_val_int,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "full_val_macro_recall": float(
                recall_score(
                    y_val_int,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "params": int(c["params"]),
            "macs": int(c["macs"]),
            "memory_mb": float(c["memory_mb"]),
        }

        (sub / "full_train_metrics.json").write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

        results.append(result)

    results.sort(
        key=lambda x: (
            -x["full_val_macro_f1"],
            x["macs"],
            x["params"],
        )
    )

    (out / "full_train_summary.json").write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
