import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

from config import (
    VAL_CSV,
    RECORDS_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR,
    LABELS,
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
        default=0,
        help="0 = toàn bộ validation"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128
    )

    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading validation data...")

    X_val, y_val, _ = build_xy(
        VAL_CSV,
        RECORDS_DIR
    )

    X_val, _ = to_keras(
        X_val,
        y_val
    )

    if (
        args.samples > 0
        and args.samples < len(X_val)
    ):
        rng = np.random.default_rng(
            SEED
        )

        indices = rng.choice(
            len(X_val),
            size=args.samples,
            replace=False
        )

        X_eval = X_val[indices]
        y_eval = y_val[indices]

    else:
        X_eval = X_val
        y_eval = y_val

    print(
        "Samples:",
        len(X_eval)
    )

    # =========================
    # QAT
    # =========================

    qat_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    print("\nLoading QAT model...")

    qat_model = tf.keras.models.load_model(
        str(qat_path),
        custom_objects={
            "QATWeightWrapper":
                QATWeightWrapper,

            "QATActivationWrapper":
                QATActivationWrapper
        },
        compile=False
    )

    print(
        "Running QAT inference..."
    )

    qat_prob = qat_model.predict(
        X_eval,
        batch_size=args.batch_size,
        verbose=1
    )

    qat_pred = np.argmax(
        qat_prob,
        axis=1
    )

    # =========================
    # INTEGER
    # =========================

    deployment_dir = (
        ARTIFACTS_DIR
        / "deployment_model"
    )

    integer_model = IntegerReferenceModel(
        deployment_dir
    )

    integer_predictions = []

    print(
        "\nRunning Integer inference..."
    )

    total = len(X_eval)

    for start in range(
        0,
        total,
        args.batch_size
    ):
        end = min(
            start + args.batch_size,
            total
        )

        result = integer_model.forward(
            X_eval[start:end]
        )

        integer_predictions.append(
            result["predictions"]
        )

        print(
            f"\rInteger: {end}/{total}",
            end=""
        )

    print()

    integer_pred = np.concatenate(
        integer_predictions
    )

    # =========================
    # REPORT
    # =========================

    print()
    print("=" * 70)
    print("QAT CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_eval,
            qat_pred,
            labels=list(
                range(
                    len(LABELS)
                )
            ),
            target_names=LABELS,
            digits=6,
            zero_division=0
        )
    )

    print()
    print("=" * 70)
    print("INTEGER CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_eval,
            integer_pred,
            labels=list(
                range(
                    len(LABELS)
                )
            ),
            target_names=LABELS,
            digits=6,
            zero_division=0
        )
    )

    agreement = (
        qat_pred
        == integer_pred
    )

    disagreement_count = int(
        np.sum(
            ~agreement
        )
    )

    print()
    print("=" * 70)
    print("PREDICTION DISAGREEMENTS")
    print("=" * 70)

    print(
        "Total:",
        len(y_eval)
    )

    print(
        "Different predictions:",
        disagreement_count
    )

    print(
        "Agreement:",
        float(
            np.mean(
                agreement
            )
        )
    )

    disagreement_by_class = {}

    for class_id, label in enumerate(
        LABELS
    ):
        class_mask = (
            y_eval
            == class_id
        )

        total_class = int(
            np.sum(
                class_mask
            )
        )

        different = int(
            np.sum(
                class_mask
                & (
                    qat_pred
                    != integer_pred
                )
            )
        )

        ratio = (
            different
            / total_class
            if total_class > 0
            else 0.0
        )

        disagreement_by_class[
            label
        ] = {
            "total":
                total_class,

            "different":
                different,

            "ratio":
                float(ratio)
        }

        print(
            f"{label}: "
            f"{different}/{total_class} "
            f"({ratio * 100:.3f}%)"
        )

    transition = confusion_matrix(
        qat_pred,
        integer_pred,
        labels=list(
            range(
                len(LABELS)
            )
        )
    )

    print()
    print(
        "QAT prediction -> Integer prediction"
    )

    print(
        transition
    )

    output = {
        "samples":
            int(
                len(y_eval)
            ),

        "qat_accuracy":
            float(
                accuracy_score(
                    y_eval,
                    qat_pred
                )
            ),

        "qat_macro_f1":
            float(
                f1_score(
                    y_eval,
                    qat_pred,
                    average="macro"
                )
            ),

        "integer_accuracy":
            float(
                accuracy_score(
                    y_eval,
                    integer_pred
                )
            ),

        "integer_macro_f1":
            float(
                f1_score(
                    y_eval,
                    integer_pred,
                    average="macro"
                )
            ),

        "agreement":
            float(
                np.mean(
                    agreement
                )
            ),

        "different_predictions":
            disagreement_count,

        "disagreement_by_true_class":
            disagreement_by_class,

        "qat_to_integer_transition":
            transition.tolist()
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_DIR
        / "integer_disagreement_analysis.json"
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