import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_train_val_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (
        X_train, y_train, y_train_int, _,
        X_val, y_val, y_val_int, _,
        X_test, y_test, y_test_int, _,
    ) = load_train_val_test(
        augment_train=False
    )

    del X_train, y_train, y_train_int
    del X_val, y_val, y_val_int

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
    )

    probs = model.predict(
        X_test,
        batch_size=256,
        verbose=1,
    )
    pred = np.argmax(probs, axis=1)

    result = {
        "test_samples": int(len(y_test_int)),
        "accuracy": float(
            accuracy_score(y_test_int, pred)
        ),
        "macro_f1": float(
            f1_score(
                y_test_int,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_test_int,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test_int,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test_int,
            pred,
            labels=[0, 1, 2, 3, 4],
        ).tolist(),
        "classification_report": classification_report(
            y_test_int,
            pred,
            target_names=["N", "L", "R", "V", "A"],
            output_dict=True,
            zero_division=0,
        ),
    }

    (out / "final_test_metrics.json").write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
