import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

from config import (
    VAL_CSV,
    RECORDS_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR,
    SEED
)

from data.loader import (
    build_xy,
    to_keras
)

from quantization.qat_layers import (
    QATWeightWrapper,
    QATActivationWrapper
)

from integer_reference.model import (
    IntegerReferenceModel
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help=(
            "Số validation samples. "
            "Dùng 0 để chạy toàn bộ."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    return parser.parse_args()


def evaluate_predictions(
    y_true,
    y_pred
):
    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro"
                )
            ),

        "weighted_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted"
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred
            ).tolist()
    }


def main():
    args = parse_args()

    np.random.seed(
        SEED
    )

    tf.random.set_seed(
        SEED
    )

    print(
        "Loading validation data..."
    )

    X_val, y_val, _ = (
        build_xy(
            VAL_CSV,
            RECORDS_DIR
        )
    )

    X_val, _ = to_keras(
        X_val,
        y_val
    )

    if (
        args.samples > 0
        and args.samples
        < len(X_val)
    ):
        rng = (
            np.random.default_rng(
                SEED
            )
        )

        indices = rng.choice(
            len(X_val),
            size=args.samples,
            replace=False
        )

        X_eval = (
            X_val[
                indices
            ]
        )

        y_eval = (
            y_val[
                indices
            ]
        )

    else:
        X_eval = X_val
        y_eval = y_val

    print(
        "Evaluation samples:",
        len(X_eval)
    )

    # ============================
    # QAT Model
    # ============================

    qat_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    print()
    print(
        "Loading QAT model..."
    )

    qat_model = (
        tf.keras.models.load_model(
            str(
                qat_path
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

    print(
        "Running QAT inference..."
    )

    start = time.perf_counter()

    qat_probabilities = (
        qat_model.predict(
            X_eval,
            batch_size=
                args.batch_size,
            verbose=1
        )
    )

    qat_time = (
        time.perf_counter()
        - start
    )

    qat_pred = np.argmax(
        qat_probabilities,
        axis=1
    )

    qat_metrics = (
        evaluate_predictions(
            y_eval,
            qat_pred
        )
    )

    # ============================
    # Integer Reference
    # ============================

    deployment_dir = (
        ARTIFACTS_DIR
        / "deployment_model"
    )

    print()
    print(
        "Loading Integer Reference Model..."
    )

    integer_model = (
        IntegerReferenceModel(
            deployment_dir
        )
    )

    integer_predictions = []

    start = time.perf_counter()

    total = len(
        X_eval
    )

    print(
        "Running integer inference..."
    )

    for start_index in range(
        0,
        total,
        args.batch_size
    ):
        end_index = min(
            start_index
            + args.batch_size,
            total
        )

        batch = (
            X_eval[
                start_index:end_index
            ]
        )

        result = (
            integer_model.forward(
                batch
            )
        )

        integer_predictions.append(
            result[
                "predictions"
            ]
        )

        print(
            f"\rInteger: "
            f"{end_index}/{total}",
            end=""
        )

    print()

    integer_time = (
        time.perf_counter()
        - start
    )

    integer_pred = np.concatenate(
        integer_predictions,
        axis=0
    )

    integer_metrics = (
        evaluate_predictions(
            y_eval,
            integer_pred
        )
    )

    agreement = float(
        np.mean(
            qat_pred
            == integer_pred
        )
    )

    macro_f1_drop = float(
        qat_metrics[
            "macro_f1"
        ]
        - integer_metrics[
            "macro_f1"
        ]
    )

    # ============================
    # Report
    # ============================

    print()
    print(
        "=" * 60
    )
    print(
        "QAT vs INTEGER REFERENCE"
    )
    print(
        "=" * 60
    )

    print()
    print(
        "QAT"
    )

    print(
        "Accuracy   :",
        qat_metrics[
            "accuracy"
        ]
    )

    print(
        "Macro F1   :",
        qat_metrics[
            "macro_f1"
        ]
    )

    print(
        "Weighted F1:",
        qat_metrics[
            "weighted_f1"
        ]
    )

    print()
    print(
        "INTEGER"
    )

    print(
        "Accuracy   :",
        integer_metrics[
            "accuracy"
        ]
    )

    print(
        "Macro F1   :",
        integer_metrics[
            "macro_f1"
        ]
    )

    print(
        "Weighted F1:",
        integer_metrics[
            "weighted_f1"
        ]
    )

    print()
    print(
        "Prediction agreement:",
        agreement
    )

    print(
        "Macro F1 drop:",
        macro_f1_drop
    )

    print()
    print(
        "QAT time:",
        qat_time,
        "seconds"
    )

    print(
        "Integer Python time:",
        integer_time,
        "seconds"
    )

    print()
    print(
        "Lưu ý: thời gian Python integer "
        "không phải latency FPGA."
    )

    output = {
        "samples":
            int(
                len(
                    X_eval
                )
            ),

        "qat":
            qat_metrics,

        "integer":
            integer_metrics,

        "prediction_agreement":
            agreement,

        "macro_f1_drop":
            macro_f1_drop,

        "qat_time_seconds":
            float(
                qat_time
            ),

        "integer_python_time_seconds":
            float(
                integer_time
            )
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_DIR
        / "integer_reference_validation.json"
    )

    with Path(
        output_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()