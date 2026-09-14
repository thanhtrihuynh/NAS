import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from config import BATCH_SIZE, RESULTS_DIR
from models.nas_model import build_nas_model
from nas.search_space import KERNEL_PATTERNS, CHANNEL_CHOICES, config_to_jsonable
from nas.cost import estimate_macs
from training.train import compile_model


def sample_architecture():
    return {
        "block1_kernels": random.choice(KERNEL_PATTERNS),
        "block1_channels": random.choice(CHANNEL_CHOICES["block1"]),
        "block2_kernels": random.choice(KERNEL_PATTERNS),
        "block2_channels": random.choice(CHANNEL_CHOICES["block2"]),
        "block3_kernels": random.choice(KERNEL_PATTERNS),
        "block3_channels": random.choice(CHANNEL_CHOICES["block3"]),
        "block4_kernels": random.choice(KERNEL_PATTERNS),
        "block4_channels": random.choice(CHANNEL_CHOICES["block4"]),
    }


def evaluate_architecture(
    config,
    X_train,
    y_train,
    X_val,
    y_val,
    search_epochs=5,
):
    model = build_nas_model(config)
    compile_model(model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=search_epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=0,
    )

    y_prob = model.predict(X_val, batch_size=512, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_val, axis=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "macro_f1": float(macro_f1),
        "params": int(model.count_params()),
        "macs": int(estimate_macs(model)),
    }


def rank_hardware_aware(results, param_weight=0.02, mac_weight=0.03):
    max_params = max(item["params"] for item in results)
    max_macs = max(item["macs"] for item in results)

    for item in results:
        param_cost = item["params"] / max_params
        mac_cost = item["macs"] / max_macs
        item["score"] = (
            item["macro_f1"]
            - param_weight * param_cost
            - mac_weight * mac_cost
        )

    return sorted(results, key=lambda x: x["score"], reverse=True)


def save_search_results(results, output_dir=RESULTS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "architecture_search_results.json"
    csv_path = output_dir / "architecture_search_results.csv"

    jsonable = []
    rows = []

    for item in results:
        record = dict(item)
        record["config"] = config_to_jsonable(record["config"])
        jsonable.append(record)

        row = {
            "id": record["id"],
            "macro_f1": record["macro_f1"],
            "params": record["params"],
            "macs": record["macs"],
            "score": record.get("score", np.nan),
            "config": json.dumps(record["config"], ensure_ascii=False),
        }
        rows.append(row)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable, f, indent=2, ensure_ascii=False)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return json_path, csv_path


def run_random_search(
    X_train,
    y_train,
    X_val,
    y_val,
    num_candidates=20,
    search_epochs=5,
):
    architecture_results = []

    for candidate_id in range(num_candidates):
        config = sample_architecture()

        print(f"\nCandidate {candidate_id + 1}/{num_candidates}")
        print(config)

        result = evaluate_architecture(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
            search_epochs=search_epochs,
        )

        print("Macro F1:", round(result["macro_f1"], 4))
        print("Params:", f"{result['params']:,}")
        print("MACs:", f"{result['macs']:,}")

        architecture_results.append(
            {
                "id": candidate_id,
                "config": config,
                **result,
            }
        )

        tf.keras.backend.clear_session()

    ranked = rank_hardware_aware(architecture_results)
    save_search_results(ranked)
    return ranked
