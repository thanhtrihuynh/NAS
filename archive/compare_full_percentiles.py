import json
from pathlib import Path

from config import (
    RESULTS_DIR,
    LABELS
)


FILES = {
    "99.95": (
        RESULTS_DIR
        / "integer_reference_p99_95_full.json"
    ),

    "99.90": (
        RESULTS_DIR
        / "integer_reference_p99_9_full.json"
    ),

    "99.50": (
        RESULTS_DIR
        / "integer_reference_p99_5_full.json"
    )
}


def class_recalls(
    confusion
):
    recalls = {}

    for index, label in enumerate(
        LABELS
    ):
        row = confusion[index]

        total = sum(row)

        correct = row[index]

        recall = (
            correct / total
            if total > 0
            else 0.0
        )

        recalls[label] = float(
            recall
        )

    return recalls


def main():
    results = []

    for percentile, path in (
        FILES.items()
    ):
        if not path.exists():
            print(
                "Missing:",
                path
            )

            continue

        with Path(
            path
        ).open(
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        qat = data[
            "qat"
        ]

        integer = data[
            "integer"
        ]

        recalls = class_recalls(
            integer[
                "confusion_matrix"
            ]
        )

        results.append({
            "percentile":
                percentile,

            "qat_macro_f1":
                float(
                    qat[
                        "macro_f1"
                    ]
                ),

            "integer_accuracy":
                float(
                    integer[
                        "accuracy"
                    ]
                ),

            "integer_macro_f1":
                float(
                    integer[
                        "macro_f1"
                    ]
                ),

            "weighted_f1":
                float(
                    integer[
                        "weighted_f1"
                    ]
                ),

            "f1_drop":
                float(
                    data[
                        "macro_f1_drop"
                    ]
                ),

            "agreement":
                float(
                    data[
                        "prediction_agreement"
                    ]
                ),

            "recalls":
                recalls
        })

    if not results:
        print(
            "Không có kết quả."
        )

        return

    results = sorted(
        results,
        key=lambda item:
            item[
                "f1_drop"
            ]
    )

    print()
    print(
        "=" * 125
    )

    print(
        "FULL VALIDATION "
        "PERCENTILE COMPARISON"
    )

    print(
        "=" * 125
    )

    print(
        f"{'Pctl':<8}"
        f"{'Accuracy':<12}"
        f"{'Macro F1':<12}"
        f"{'F1 drop':<12}"
        f"{'Agreement':<12}"
        f"{'N rec':<10}"
        f"{'L rec':<10}"
        f"{'R rec':<10}"
        f"{'V rec':<10}"
        f"{'A rec':<10}"
    )

    print(
        "-" * 125
    )

    for item in results:
        recalls = item[
            "recalls"
        ]

        print(
            f"{item['percentile']:<8}"
            f"{item['integer_accuracy']:<12.6f}"
            f"{item['integer_macro_f1']:<12.6f}"
            f"{item['f1_drop']:<12.6f}"
            f"{item['agreement']:<12.6f}"
            f"{recalls['N']:<10.6f}"
            f"{recalls['L']:<10.6f}"
            f"{recalls['R']:<10.6f}"
            f"{recalls['V']:<10.6f}"
            f"{recalls['A']:<10.6f}"
        )

    best = results[0]

    print()
    print(
        "=" * 125
    )

    print(
        "BEST CONFIGURATION"
    )

    print(
        "=" * 125
    )

    print(
        "Percentile:",
        best[
            "percentile"
        ]
    )

    print(
        "Integer Macro F1:",
        best[
            "integer_macro_f1"
        ]
    )

    print(
        "Macro F1 drop:",
        best[
            "f1_drop"
        ]
    )

    print(
        "Prediction agreement:",
        best[
            "agreement"
        ]
    )

    print(
        "Class A recall:",
        best[
            "recalls"
        ][
            "A"
        ]
    )


if __name__ == "__main__":
    main()