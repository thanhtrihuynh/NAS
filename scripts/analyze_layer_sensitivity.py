import gc
import json

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from config import (
    BEST_NAS_MODEL_PATH,
    VAL_CSV,
    RECORDS_DIR,
    RESULTS_DIR,
    ARTIFACTS_DIR
)

from data.loader import (
    build_xy,
    to_keras
)

from quantization.mixed_precision import (
    get_searchable_layer_names,
    build_layerwise_mixed_precision_model,
    estimate_layerwise_cost
)


SINGLE_LAYER_DROP_LIMIT = 0.005


def evaluate_model(
    model,
    X,
    y_true
):
    y_prob = model.predict(
        X,
        batch_size=512,
        verbose=0
    )

    y_pred = np.argmax(
        y_prob,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return (
        float(accuracy),
        float(macro_f1)
    )


def main():
    if not BEST_NAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: "
            f"{BEST_NAS_MODEL_PATH}"
        )

    X_val, y_val_int, _ = (
        build_xy(
            VAL_CSV,
            RECORDS_DIR
        )
    )

    X_val_keras, _ = (
        to_keras(
            X_val,
            y_val_int
        )
    )

    fp32_model = (
        tf.keras.models.load_model(
            str(
                BEST_NAS_MODEL_PATH
            ),
            compile=False
        )
    )

    searchable_layers = (
        get_searchable_layer_names(
            fp32_model
        )
    )

    print(
        "\nSố computational layers:",
        len(searchable_layers)
    )

    print(
        "\nCác layer sẽ phân tích:"
    )

    for index, name in enumerate(
        searchable_layers,
        start=1
    ):
        print(
            f"{index:02d}. {name}"
        )

    print(
        "\n=== FP32 REFERENCE ==="
    )

    fp32_accuracy, fp32_f1 = (
        evaluate_model(
            fp32_model,
            X_val_keras,
            y_val_int
        )
    )

    print(
        "Accuracy:",
        fp32_accuracy
    )

    print(
        "Macro F1:",
        fp32_f1
    )

    all_int8_config = {
        name: 8
        for name
        in searchable_layers
    }

    print(
        "\n=== ALL INT8 REFERENCE ==="
    )

    int8_model = (
        build_layerwise_mixed_precision_model(
            fp32_model,
            all_int8_config
        )
    )

    int8_accuracy, int8_f1 = (
        evaluate_model(
            int8_model,
            X_val_keras,
            y_val_int
        )
    )

    int8_cost = (
        estimate_layerwise_cost(
            fp32_model,
            all_int8_config
        )
    )

    print(
        "Accuracy:",
        int8_accuracy
    )

    print(
        "Macro F1:",
        int8_f1
    )

    print(
        "Weight memory:",
        int8_cost[
            "weight_memory_kb"
        ],
        "KB"
    )

    del int8_model
    gc.collect()

    results = []

    for index, layer_name in enumerate(
        searchable_layers,
        start=1
    ):
        print(
            "\n================================"
        )

        print(
            f"Layer {index}/"
            f"{len(searchable_layers)}"
        )

        print(
            "INT4:",
            layer_name
        )

        config = dict(
            all_int8_config
        )

        config[
            layer_name
        ] = 4

        mixed_model = (
            build_layerwise_mixed_precision_model(
                fp32_model,
                config
            )
        )

        accuracy, macro_f1 = (
            evaluate_model(
                mixed_model,
                X_val_keras,
                y_val_int
            )
        )

        cost = (
            estimate_layerwise_cost(
                fp32_model,
                config
            )
        )

        f1_drop_vs_int8 = (
            int8_f1
            - macro_f1
        )

        f1_drop_vs_fp32 = (
            fp32_f1
            - macro_f1
        )

        sensitive = (
            f1_drop_vs_int8
            > SINGLE_LAYER_DROP_LIMIT
        )

        memory_ratio = (
            cost[
                "weight_memory_kb"
            ]
            /
            int8_cost[
                "weight_memory_kb"
            ]
        )

        bitops_ratio = (
            cost["bitops"]
            /
            int8_cost["bitops"]
        )

        result = {
            "layer":
                layer_name,

            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "f1_drop_vs_int8":
                f1_drop_vs_int8,

            "f1_drop_vs_fp32":
                f1_drop_vs_fp32,

            "weight_memory_kb":
                cost[
                    "weight_memory_kb"
                ],

            "memory_ratio":
                memory_ratio,

            "bitops_ratio":
                bitops_ratio,

            "sensitive":
                sensitive
        }

        results.append(
            result
        )

        print(
            "Accuracy:",
            accuracy
        )

        print(
            "Macro F1:",
            macro_f1
        )

        print(
            "Drop vs INT8:",
            f1_drop_vs_int8
        )

        print(
            "Memory ratio:",
            memory_ratio
        )

        print(
            "BitOps ratio:",
            bitops_ratio
        )

        print(
            "Sensitive:",
            sensitive
        )

        del mixed_model
        gc.collect()

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df.sort_values(
            by="f1_drop_vs_int8",
            ascending=False
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        RESULTS_DIR
        / "layer_sensitivity.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    locked_int8_layers = (
        results_df[
            results_df[
                "sensitive"
            ]
        ]["layer"]
        .tolist()
    )

    candidate_layers = (
        results_df[
            ~results_df[
                "sensitive"
            ]
        ]["layer"]
        .tolist()
    )

    json_output = {
        "fp32_reference": {
            "accuracy":
                fp32_accuracy,

            "macro_f1":
                fp32_f1
        },

        "int8_reference": {
            "accuracy":
                int8_accuracy,

            "macro_f1":
                int8_f1,

            "weight_memory_kb":
                int8_cost[
                    "weight_memory_kb"
                ],

            "bitops":
                int8_cost[
                    "bitops"
                ]
        },

        "single_layer_drop_limit":
            SINGLE_LAYER_DROP_LIMIT,

        "searchable_layers":
            searchable_layers,

        "locked_int8_layers":
            locked_int8_layers,

        "candidate_layers":
            candidate_layers,

        "results":
            results
    }

    json_path = (
        ARTIFACTS_DIR
        / "layer_sensitivity.json"
    )

    with Path(
        json_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            json_output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\n================================"
    )

    print(
        "LAYER SENSITIVITY FINISHED"
    )

    print(
        "================================"
    )

    print(
        "\nLocked INT8 layers:",
        len(
            locked_int8_layers
        )
    )

    for name in locked_int8_layers:
        print(
            "INT8  ",
            name
        )

    print(
        "\nCó thể search INT4/INT8:",
        len(
            candidate_layers
        )
    )

    for name in candidate_layers:
        print(
            "4/8   ",
            name
        )

    print(
        "\nSaved:"
    )

    print(
        csv_path
    )

    print(
        json_path
    )


if __name__ == "__main__":
    main()