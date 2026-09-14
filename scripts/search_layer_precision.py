import gc
import json
import random

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
    ARTIFACTS_DIR,
    SEED
)

from data.loader import (
    build_xy,
    to_keras
)

from quantization.mixed_precision import (
    build_layerwise_mixed_precision_model,
    estimate_layerwise_cost,
    config_signature
)


POPULATION_SIZE = 16
ELITE_SIZE = 4
GENERATIONS = 5

MUTATION_RATE = 0.15

MAX_F1_DROP = 0.005


random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def evaluate_predictions(
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


def all_int8_config(
    layer_names
):
    return {
        name: 8
        for name
        in layer_names
    }


def all_candidate_int4_config(
    layer_names,
    candidate_layers,
    locked_layers
):
    config = all_int8_config(
        layer_names
    )

    for name in candidate_layers:
        config[name] = 4

    for name in locked_layers:
        config[name] = 8

    return config


def random_config(
    layer_names,
    candidate_layers,
    locked_layers
):
    config = all_int8_config(
        layer_names
    )

    for name in candidate_layers:
        config[name] = random.choice(
            [4, 8]
        )

    for name in locked_layers:
        config[name] = 8

    return config


def crossover(
    parent_a,
    parent_b,
    layer_names,
    candidate_layers,
    locked_layers
):
    child = all_int8_config(
        layer_names
    )

    for name in candidate_layers:
        child[name] = random.choice(
            [
                parent_a[name],
                parent_b[name]
            ]
        )

    for name in locked_layers:
        child[name] = 8

    return child


def mutate(
    config,
    candidate_layers,
    locked_layers
):
    child = dict(
        config
    )

    changed = False

    for name in candidate_layers:
        if (
            random.random()
            < MUTATION_RATE
        ):
            child[name] = (
                4
                if child[name] == 8
                else 8
            )

            changed = True

    # Bảo đảm có mutation
    if (
        not changed
        and len(candidate_layers) > 0
    ):
        name = random.choice(
            candidate_layers
        )

        child[name] = (
            4
            if child[name] == 8
            else 8
        )

    for name in locked_layers:
        child[name] = 8

    return child


def ranking_key(result):
    if result["accepted"]:
        return (
            2,
            -result[
                "hardware_cost"
            ],
            result[
                "macro_f1"
            ]
        )

    return (
        1,
        result[
            "macro_f1"
        ],
        -result[
            "hardware_cost"
        ]
    )


def main():
    sensitivity_path = (
        ARTIFACTS_DIR
        / "layer_sensitivity.json"
    )

    if not sensitivity_path.exists():
        raise FileNotFoundError(
            "Chưa có layer_sensitivity.json.\n"
            "Hãy chạy trước:\n"
            "python -m "
            "scripts.analyze_layer_sensitivity"
        )

    with sensitivity_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        sensitivity = json.load(
            f
        )

    layer_names = (
        sensitivity[
            "searchable_layers"
        ]
    )

    locked_layers = (
        sensitivity[
            "locked_int8_layers"
        ]
    )

    candidate_layers = (
        sensitivity[
            "candidate_layers"
        ]
    )

    if len(candidate_layers) == 0:
        raise RuntimeError(
            "Không có layer nào đủ ổn định "
            "để dùng PTQ INT4.\n"
            "Bước tiếp theo phải dùng QAT."
        )

    print(
        "Total computational layers:",
        len(layer_names)
    )

    print(
        "Locked INT8:",
        len(locked_layers)
    )

    print(
        "Search INT4/INT8:",
        len(candidate_layers)
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

    fp32_accuracy, fp32_f1 = (
        evaluate_predictions(
            fp32_model,
            X_val_keras,
            y_val_int
        )
    )

    minimum_f1 = (
        fp32_f1
        - MAX_F1_DROP
    )

    print(
        "\nFP32 Macro F1:",
        fp32_f1
    )

    print(
        "Minimum accepted F1:",
        minimum_f1
    )

    reference_config = (
        all_int8_config(
            layer_names
        )
    )

    reference_cost = (
        estimate_layerwise_cost(
            fp32_model,
            reference_config
        )
    )

    evaluation_cache = {}

    def evaluate_config(config):
        signature = (
            config_signature(
                config,
                layer_names
            )
        )

        if signature in (
            evaluation_cache
        ):
            return (
                evaluation_cache[
                    signature
                ]
            )

        model = (
            build_layerwise_mixed_precision_model(
                fp32_model,
                config
            )
        )

        accuracy, macro_f1 = (
            evaluate_predictions(
                model,
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

        memory_ratio = (
            cost[
                "weight_memory_kb"
            ]
            /
            reference_cost[
                "weight_memory_kb"
            ]
        )

        bitops_ratio = (
            cost["bitops"]
            /
            reference_cost["bitops"]
        )

        hardware_cost = (
            0.5
            * memory_ratio
            +
            0.5
            * bitops_ratio
        )

        f1_drop = (
            fp32_f1
            - macro_f1
        )

        accepted = (
            macro_f1
            >= minimum_f1
        )

        int4_layers = [
            name
            for name
            in layer_names
            if config[name] == 4
        ]

        result = {
            "accuracy":
                accuracy,

            "macro_f1":
                macro_f1,

            "f1_drop":
                f1_drop,

            "weight_memory_kb":
                cost[
                    "weight_memory_kb"
                ],

            "memory_ratio":
                memory_ratio,

            "bitops":
                cost[
                    "bitops"
                ],

            "bitops_ratio":
                bitops_ratio,

            "hardware_cost":
                hardware_cost,

            "accepted":
                accepted,

            "int4_count":
                len(
                    int4_layers
                ),

            "int8_count":
                (
                    len(layer_names)
                    - len(
                        int4_layers
                    )
                ),

            "config":
                dict(config)
        }

        evaluation_cache[
            signature
        ] = result

        del model
        gc.collect()

        return result

    # Population ban đầu
    population = []

    population.append(
        all_int8_config(
            layer_names
        )
    )

    population.append(
        all_candidate_int4_config(
            layer_names,
            candidate_layers,
            locked_layers
        )
    )

    # Thêm một số candidate
    # dựa trên layer ít nhạy
    sensitivity_results = (
        sensitivity["results"]
    )

    sorted_sensitivity = sorted(
        sensitivity_results,
        key=lambda x:
            x[
                "f1_drop_vs_int8"
            ]
    )

    for item in sorted_sensitivity:
        name = item["layer"]

        if (
            name
            not in candidate_layers
        ):
            continue

        config = (
            all_int8_config(
                layer_names
            )
        )

        config[name] = 4

        population.append(
            config
        )

        if len(population) >= 6:
            break

    while (
        len(population)
        < POPULATION_SIZE
    ):
        population.append(
            random_config(
                layer_names,
                candidate_layers,
                locked_layers
            )
        )

    for generation in range(
        1,
        GENERATIONS + 1
    ):
        print(
            "\n================================"
        )

        print(
            f"GENERATION "
            f"{generation}/"
            f"{GENERATIONS}"
        )

        print(
            "================================"
        )

        evaluated = []

        for index, config in enumerate(
            population,
            start=1
        ):
            result = (
                evaluate_config(
                    config
                )
            )

            evaluated.append(
                result
            )

            print(
                f"{index:02d}/"
                f"{len(population)}",
                "| F1 =",
                round(
                    result[
                        "macro_f1"
                    ],
                    6
                ),
                "| INT4 =",
                result[
                    "int4_count"
                ],
                "| Mem =",
                round(
                    result[
                        "memory_ratio"
                    ],
                    4
                ),
                "| BitOps =",
                round(
                    result[
                        "bitops_ratio"
                    ],
                    4
                ),
                "| Accepted =",
                result[
                    "accepted"
                ]
            )

        evaluated.sort(
            key=ranking_key,
            reverse=True
        )

        best_generation = (
            evaluated[0]
        )

        print(
            "\nBest generation:"
        )

        print(
            "Macro F1:",
            best_generation[
                "macro_f1"
            ]
        )

        print(
            "INT4 layers:",
            best_generation[
                "int4_count"
            ]
        )

        print(
            "Memory ratio:",
            best_generation[
                "memory_ratio"
            ]
        )

        print(
            "BitOps ratio:",
            best_generation[
                "bitops_ratio"
            ]
        )

        # Giữ elite
        elites = [
            dict(
                result["config"]
            )
            for result
            in evaluated[
                :ELITE_SIZE
            ]
        ]

        next_population = list(
            elites
        )

        while (
            len(next_population)
            < POPULATION_SIZE
        ):
            parent_a = (
                random.choice(
                    elites
                )
            )

            parent_b = (
                random.choice(
                    elites
                )
            )

            child = crossover(
                parent_a,
                parent_b,
                layer_names,
                candidate_layers,
                locked_layers
            )

            child = mutate(
                child,
                candidate_layers,
                locked_layers
            )

            next_population.append(
                child
            )

        population = (
            next_population
        )

    all_results = list(
        evaluation_cache.values()
    )

    accepted_results = [
        result
        for result
        in all_results
        if result[
            "accepted"
        ]
    ]

    if len(
        accepted_results
    ) > 0:
        accepted_results.sort(
            key=lambda x: (
                x[
                    "hardware_cost"
                ],
                -x[
                    "macro_f1"
                ]
            )
        )

        best = (
            accepted_results[0]
        )

        selection_reason = (
            "lowest_hardware_cost_"
            "within_f1_constraint"
        )

    else:
        all_results.sort(
            key=lambda x:
                x[
                    "macro_f1"
                ],
            reverse=True
        )

        best = (
            all_results[0]
        )

        selection_reason = (
            "no_candidate_met_"
            "f1_constraint"
        )

    rows = []

    for index, result in enumerate(
        all_results,
        start=1
    ):
        rows.append({
            "candidate":
                index,

            "accuracy":
                result[
                    "accuracy"
                ],

            "macro_f1":
                result[
                    "macro_f1"
                ],

            "f1_drop":
                result[
                    "f1_drop"
                ],

            "int4_count":
                result[
                    "int4_count"
                ],

            "int8_count":
                result[
                    "int8_count"
                ],

            "weight_memory_kb":
                result[
                    "weight_memory_kb"
                ],

            "memory_ratio":
                result[
                    "memory_ratio"
                ],

            "bitops_ratio":
                result[
                    "bitops_ratio"
                ],

            "hardware_cost":
                result[
                    "hardware_cost"
                ],

            "accepted":
                result[
                    "accepted"
                ],

            "config":
                json.dumps(
                    result[
                        "config"
                    ],
                    ensure_ascii=False
                )
        })

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        RESULTS_DIR
        / "layer_precision_search.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        csv_path,
        index=False
    )

    output = {
        "selection_reason":
            selection_reason,

        "fp32_reference": {
            "accuracy":
                fp32_accuracy,

            "macro_f1":
                fp32_f1
        },

        "max_allowed_f1_drop":
            MAX_F1_DROP,

        "minimum_macro_f1":
            minimum_f1,

        "accuracy":
            best[
                "accuracy"
            ],

        "macro_f1":
            best[
                "macro_f1"
            ],

        "f1_drop":
            best[
                "f1_drop"
            ],

        "int4_count":
            best[
                "int4_count"
            ],

        "int8_count":
            best[
                "int8_count"
            ],

        "weight_memory_kb":
            best[
                "weight_memory_kb"
            ],

        "memory_ratio":
            best[
                "memory_ratio"
            ],

        "bitops_ratio":
            best[
                "bitops_ratio"
            ],

        "hardware_cost":
            best[
                "hardware_cost"
            ],

        "precision":
            best[
                "config"
            ]
    }

    json_path = (
        ARTIFACTS_DIR
        / "best_layer_precision_config.json"
    )

    with Path(
        json_path
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

    print(
        "\n\n================================"
    )

    print(
        "BEST LAYER-WISE "
        "MIXED PRECISION"
    )

    print(
        "================================"
    )

    print(
        "Accuracy:",
        best[
            "accuracy"
        ]
    )

    print(
        "Macro F1:",
        best[
            "macro_f1"
        ]
    )

    print(
        "F1 drop:",
        best[
            "f1_drop"
        ]
    )

    print(
        "INT4 layers:",
        best[
            "int4_count"
        ]
    )

    print(
        "INT8 layers:",
        best[
            "int8_count"
        ]
    )

    print(
        "Weight memory:",
        best[
            "weight_memory_kb"
        ],
        "KB"
    )

    print(
        "Memory ratio:",
        best[
            "memory_ratio"
        ]
    )

    print(
        "BitOps ratio:",
        best[
            "bitops_ratio"
        ]
    )

    print(
        "\nPRECISION PER LAYER:"
    )

    for name in layer_names:
        bits = best[
            "config"
        ][name]

        print(
            f"{name:<30} "
            f"INT{bits}"
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