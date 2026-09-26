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
from nas.cost import estimate_macs


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to best_candidate.json",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate = json.loads(
        candidate_path.read_text(
            encoding="utf-8"
        )
    )

    architecture = candidate["architecture"]

    print("=" * 60)
    print("FULL TRAIN NAS CANDIDATE")
    print("=" * 60)
    print(f"Candidate ID : {candidate.get('id')}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Seed         : {args.seed}")
    print()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    (
        X_train,
        y_train,
        y_train_int,
        _,
        X_val,
        y_val,
        y_val_int,
        _,
        X_test,
        y_test,
        y_test_int,
        _,
    ) = load_train_val_test(
        augment_train=True,
        seed=args.seed,
    )

    sample_weight = make_sample_weights(
        y_train_int
    )

    tf.keras.backend.clear_session()

    model = build_nas_model(
        architecture
    )

    params = int(
        model.count_params()
    )

    macs = int(
        estimate_macs(model)
    )

    print(f"Params : {params:,}")
    print(f"MACs   : {macs:,}")
    print()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.01
        ),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                output_dir / "best_model.keras"
            ),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        sample_weight=sample_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Validation
    val_probs = model.predict(
        X_val,
        batch_size=256,
        verbose=1,
    )

    val_pred = np.argmax(
        val_probs,
        axis=1,
    )

    val_metrics = {
        "accuracy": float(
            accuracy_score(
                y_val_int,
                val_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_val_int,
                val_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    # Final test
    test_probs = model.predict(
        X_test,
        batch_size=256,
        verbose=1,
    )

    test_pred = np.argmax(
        test_probs,
        axis=1,
    )

    test_metrics = {
        "accuracy": float(
            accuracy_score(
                y_test_int,
                test_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_test_int,
                test_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_test_int,
                test_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test_int,
                test_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    results = {
        "candidate_id": candidate.get("id"),
        "architecture": architecture,
        "epochs_requested": args.epochs,
        "epochs_ran": len(
            history.history["loss"]
        ),
        "params": params,
        "macs": macs,
        "validation": val_metrics,
        "test": test_metrics,
    }

    save_json(
        output_dir / "metrics.json",
        results,
    )

    save_json(
        output_dir / "history.json",
        {
            k: [float(x) for x in v]
            for k, v in history.history.items()
        },
    )

    print()
    print("=" * 60)
    print("FULL TRAIN COMPLETED")
    print("=" * 60)

    print(
        f"Val Macro F1 : "
        f"{val_metrics['macro_f1']:.6f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()