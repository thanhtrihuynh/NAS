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
    BEST_NAS_MODEL_PATH,
    TRAIN_CSV,
    VAL_CSV,
    RECORDS_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    SEED
)

from data.loader import (
    build_xy,
    to_keras
)

from quantization.qat_layers import (
    QATWeightWrapper,
    QATActivationWrapper,
    build_qat_model
)

from training.train_qat import (
    train_qat_model
)


QAT_EPOCHS = 15
QAT_LEARNING_RATE = 1e-5


def evaluate(
    model,
    X,
    y_true,
    title
):
    print(
        "\n=============================="
    )

    print(
        title
    )

    print(
        "=============================="
    )

    probabilities = (
        model.predict(
            X,
            batch_size=512,
            verbose=0
        )
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    print(
        "Accuracy:",
        accuracy
    )

    print(
        "Macro F1:",
        macro_f1
    )

    return {
        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "predictions":
            predictions
    }


def main():
    np.random.seed(
        SEED
    )

    tf.random.set_seed(
        SEED
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    precision_path = (
        ARTIFACTS_DIR
        /
        "best_layer_precision_config.json"
    )

    if not (
        precision_path.exists()
    ):
        raise FileNotFoundError(
            "Không tìm thấy "
            "best_layer_precision_config.json"
        )

    if not (
        BEST_NAS_MODEL_PATH.exists()
    ):
        raise FileNotFoundError(
            f"Không tìm thấy "
            f"{BEST_NAS_MODEL_PATH}"
        )

    with Path(
        precision_path
    ).open(
        "r",
        encoding="utf-8"
    ) as f:
        search_result = (
            json.load(f)
        )

    precision_config = (
        search_result[
            "precision"
        ]
    )

    print(
        "\n=== PRECISION CONFIG ==="
    )

    int4_layers = []
    int8_layers = []

    for layer_name, bits in (
        precision_config.items()
    ):
        print(
            f"{layer_name:<30} "
            f"INT{bits}"
        )

        if bits == 4:
            int4_layers.append(
                layer_name
            )
        else:
            int8_layers.append(
                layer_name
            )

    print(
        "\nINT4 layers:",
        len(int4_layers)
    )

    print(
        "INT8 layers:",
        len(int8_layers)
    )

    print(
        "\nLoading TRAIN..."
    )

    X_train, y_train_int, _ = (
        build_xy(
            TRAIN_CSV,
            RECORDS_DIR
        )
    )

    print(
        "Loading VALIDATION..."
    )

    X_val, y_val_int, _ = (
        build_xy(
            VAL_CSV,
            RECORDS_DIR
        )
    )

    X_train_keras, y_train_keras = (
        to_keras(
            X_train,
            y_train_int
        )
    )

    X_val_keras, y_val_keras = (
        to_keras(
            X_val,
            y_val_int
        )
    )

    print(
        "\nTrain shape:",
        X_train_keras.shape
    )

    print(
        "Validation shape:",
        X_val_keras.shape
    )

    print(
        "\nLoading FP32 NAS model..."
    )

    fp32_model = (
        tf.keras.models.load_model(
            str(
                BEST_NAS_MODEL_PATH
            ),
            compile=False
        )
    )

    fp32_result = evaluate(
        fp32_model,
        X_val_keras,
        y_val_int,
        "FP32 REFERENCE"
    )

    print(
        "\nBuilding QAT model..."
    )

    qat_model = (
        build_qat_model(
            fp32_model,
            precision_config
        )
    )

    print(
        "\nQAT model parameters:",
        qat_model.count_params()
    )

    before_result = evaluate(
        qat_model,
        X_val_keras,
        y_val_int,
        "MIXED PRECISION BEFORE QAT"
    )

    checkpoint_path = (
        ARTIFACTS_DIR
        /
        "best_mixed_precision_qat.keras"
    )

    print(
        "\n=============================="
    )

    print(
        "START QAT"
    )

    print(
        "=============================="
    )

    history = (
        train_qat_model(
            qat_model,
            X_train_keras,
            y_train_keras,
            X_val_keras,
            y_val_keras,
            checkpoint_path=
                checkpoint_path,
            epochs=
                QAT_EPOCHS,
            batch_size=
                BATCH_SIZE,
            learning_rate=
                QAT_LEARNING_RATE
        )
    )

    history_path = (
        RESULTS_DIR
        /
        "qat_training_history.csv"
    )

    pd.DataFrame(
        history.history
    ).to_csv(
        history_path,
        index=False
    )

    print(
        "\nLoading best QAT checkpoint..."
    )

    best_qat_model = (
        tf.keras.models.load_model(
            str(
                checkpoint_path
            ),
            custom_objects={
                "QATWeightWrapper":
                    QATWeightWrapper,

                "QATActivationWrapper":
                    QATActivationWrapper
            },
            compile=False
        )
    )

    final_result = evaluate(
        best_qat_model,
        X_val_keras,
        y_val_int,
        "BEST QAT VALIDATION"
    )

    predictions = (
        final_result[
            "predictions"
        ]
    )

    report = (
        classification_report(
            y_val_int,
            predictions,
            digits=6,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_val_int,
        predictions
    )

    print(
        "\nClassification report:"
    )

    print(
        report
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        cm
    )

    report_path = (
        RESULTS_DIR
        /
        "qat_classification_report.txt"
    )

    with Path(
        report_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            report
        )

    cm_path = (
        RESULTS_DIR
        /
        "qat_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm
    ).to_csv(
        cm_path,
        index=False
    )

    metrics = {
        "fp32": {
            "accuracy":
                fp32_result[
                    "accuracy"
                ],

            "macro_f1":
                fp32_result[
                    "macro_f1"
                ]
        },

        "mixed_before_qat": {
            "accuracy":
                before_result[
                    "accuracy"
                ],

            "macro_f1":
                before_result[
                    "macro_f1"
                ]
        },

        "mixed_after_qat": {
            "accuracy":
                final_result[
                    "accuracy"
                ],

            "macro_f1":
                final_result[
                    "macro_f1"
                ]
        },

        "int4_layers":
            int4_layers,

        "int8_layers":
            int8_layers,

        "qat_epochs":
            QAT_EPOCHS,

        "qat_learning_rate":
            QAT_LEARNING_RATE
    }

    metrics_path = (
        RESULTS_DIR
        /
        "qat_validation_metrics.json"
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
            indent=2,
            ensure_ascii=False
        )

    print(
        "\n=============================="
    )

    print(
        "QAT FINISHED"
    )

    print(
        "=============================="
    )

    print(
        "\nFP32 Macro F1:",
        fp32_result[
            "macro_f1"
        ]
    )

    print(
        "Before QAT Macro F1:",
        before_result[
            "macro_f1"
        ]
    )

    print(
        "After QAT Macro F1:",
        final_result[
            "macro_f1"
        ]
    )

    print(
        "\nSaved model:"
    )

    print(
        checkpoint_path
    )

    print(
        "\nSaved history:"
    )

    print(
        history_path
    )

    print(
        "\nSaved metrics:"
    )

    print(
        metrics_path
    )


if __name__ == "__main__":
    main()