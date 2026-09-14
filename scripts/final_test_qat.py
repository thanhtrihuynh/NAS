import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import (
    TEST_CSV,
    RECORDS_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR,
    LABELS
)

from data.loader import (
    build_xy,
    to_keras
)

from quantization.qat_layers import (
    QATWeightWrapper,
    QATActivationWrapper
)


def main():
    model_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {model_path}"
        )

    print("Loading TEST set...")

    X_test, y_test, _ = build_xy(
        TEST_CSV,
        RECORDS_DIR
    )

    X_test, _ = to_keras(
        X_test,
        y_test
    )

    print("Test shape:", X_test.shape)

    print("\nLoading final QAT model...")

    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "QATWeightWrapper":
                QATWeightWrapper,
            "QATActivationWrapper":
                QATActivationWrapper
        },
        compile=False
    )

    print("\nRunning final inference...")

    probabilities = model.predict(
        X_test,
        batch_size=512,
        verbose=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n==============================")
    print("FINAL TEST RESULT")
    print("==============================")

    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    report = classification_report(
        y_test,
        predictions,
        target_names=LABELS,
        digits=6,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        RESULTS_DIR
        / "final_test_classification_report.txt"
    )

    with Path(
        report_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    cm_path = (
        RESULTS_DIR
        / "final_test_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=LABELS,
        columns=LABELS
    ).to_csv(
        cm_path
    )

    metrics = {
        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "weighted_f1":
            float(weighted_f1),

        "num_test_samples":
            int(len(y_test))
    }

    metrics_path = (
        RESULTS_DIR
        / "final_test_metrics.json"
    )

    with Path(
        metrics_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    predictions_path = (
        RESULTS_DIR
        / "final_test_predictions.csv"
    )

    pd.DataFrame({
        "true_label_id":
            y_test,

        "predicted_label_id":
            predictions,

        "true_label": [
            LABELS[i]
            for i in y_test
        ],

        "predicted_label": [
            LABELS[i]
            for i in predictions
        ],

        "confidence":
            np.max(
                probabilities,
                axis=1
            )
    }).to_csv(
        predictions_path,
        index=False
    )

    print("\nSaved:")
    print(metrics_path)
    print(report_path)
    print(cm_path)
    print(predictions_path)


if __name__ == "__main__":
    main()