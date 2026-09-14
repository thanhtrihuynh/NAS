import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


# Ensure Vietnamese log messages can be written on Windows even when
# the process is redirected to stdout.log/stderr.log.
try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    pass

try:
    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace",
    )
except Exception:
    pass


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def pick_kernel_triplet(kernels):
    values = sorted(
        {
            int(value)
            for value in kernels
        }
    )

    if len(values) < 3:
        raise ValueError(
            "Mỗi Inception block cần 3 kernel branches. "
            "Search space phải có ít nhất 3 kernel."
        )

    return tuple(
        sorted(
            random.sample(
                values,
                3,
            )
        )
    )


def pick_channels(channels):
    values = sorted(
        {
            int(value)
            for value in channels
            if int(value) > 0
        }
    )

    if not values:
        raise ValueError(
            "Channel Candidates đang rỗng."
        )

    # Four blocks. Sorting keeps width non-decreasing with depth.
    return sorted(
        random.choices(
            values,
            k=4,
        )
    )


def sample_architecture(config):
    channels = pick_channels(
        config["channels"]
    )

    return {
        "block1_kernels":
            pick_kernel_triplet(
                config["kernels"]
            ),

        "block1_channels":
            channels[0],

        "block2_kernels":
            pick_kernel_triplet(
                config["kernels"]
            ),

        "block2_channels":
            channels[1],

        "block3_kernels":
            pick_kernel_triplet(
                config["kernels"]
            ),

        "block3_channels":
            channels[2],

        "block4_kernels":
            pick_kernel_triplet(
                config["kernels"]
            ),

        "block4_channels":
            channels[3],
    }


def normalize_loaded_data(data):
    if isinstance(data, dict):
        required = [
            "X_train",
            "y_train",
            "X_val",
            "y_val",
        ]

        missing = [
            key
            for key in required
            if key not in data
        ]

        if missing:
            raise KeyError(
                "load_train_val_test() thiếu keys: "
                + ", ".join(missing)
            )

        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
        )

    if isinstance(data, (tuple, list)) and len(data) >= 4:
        return (
            data[0],
            data[1],
            data[2],
            data[3],
        )

    raise TypeError(
        "load_train_val_test() phải trả dict hoặc tuple/list >= 4 phần tử."
    )


def build_candidate_model(build_nas_model, arch):
    try:
        return build_nas_model(
            arch
        )
    except TypeError as first_error:
        try:
            return build_nas_model(
                **arch
            )
        except TypeError:
            raise first_error


def compile_candidate(compile_model, model):
    result = compile_model(
        model
    )

    # Support compile_model(model) functions that either return the model
    # or mutate/compile in place and return None.
    return (
        result
        if result is not None
        else model
    )


def select_best(results, objective):
    candidates = [
        item
        for item in results
        if (
            item.get("status")
            == "completed"
            and item.get("macro_f1")
            is not None
        )
    ]

    if not candidates:
        return None

    best_f1 = max(
        item["macro_f1"]
        for item in candidates
    )

    max_params = max(
        max(
            int(item["params"]),
            1,
        )
        for item in candidates
    )

    max_macs = max(
        max(
            int(item["macs"]),
            1,
        )
        for item in candidates
    )

    if objective == "macro_f1":
        for item in candidates:
            item["score"] = float(
                item["macro_f1"]
            )

        ranked = sorted(
            candidates,
            key=lambda item: (
                item["macro_f1"],
                -item["macs"],
                -item["params"],
            ),
            reverse=True,
        )

        reason = (
            "highest_macro_f1"
        )

    elif objective == "efficient":
        threshold = (
            best_f1
            - 0.005
        )

        pool = [
            item
            for item in candidates
            if (
                item["macro_f1"]
                >= threshold
            )
        ]

        for item in candidates:
            item["score"] = float(
                item["macro_f1"]
                - item["macs"]
                / max_macs
            )

        ranked = sorted(
            pool,
            key=lambda item: (
                item["macs"],
                item["params"],
                -item["macro_f1"],
            ),
        )

        reason = (
            "lowest_macs_within_0.005_f1_of_best"
        )

    else:
        for item in candidates:
            item["score"] = float(
                item["macro_f1"]
                - 0.02
                * item["params"]
                / max_params
                - 0.03
                * item["macs"]
                / max_macs
            )

        ranked = sorted(
            candidates,
            key=lambda item:
                item["score"],
            reverse=True,
        )

        reason = (
            "balanced_macro_f1_params_macs"
        )

    best = ranked[0]
    best["selection_reason"] = reason

    return best


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_json(
        args.config
    )

    trials = max(
        1,
        int(
            config.get(
                "trials",
                20,
            )
        ),
    )

    epochs = max(
        1,
        int(
            config.get(
                "epochs_per_candidate",
                5,
            )
        ),
    )

    objective = str(
        config.get(
            "objective",
            "balanced",
        )
    )

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    project_root = Path.cwd()

    if str(project_root) not in sys.path:
        sys.path.insert(
            0,
            str(project_root),
        )

    save_json(
        output_dir / "progress.json",
        {
            "status":
                "importing_project_modules",

            "completed":
                0,

            "total":
                trials,
        },
    )

    # These imports intentionally use the existing project implementation.
    from data.loader import load_train_val_test
    from models.nas_model import build_nas_model
    from nas.cost import estimate_macs
    from training.train import compile_model

    save_json(
        output_dir / "progress.json",
        {
            "status":
                "loading_train_val",

            "completed":
                0,

            "total":
                trials,
        },
    )

    try:
        loaded = load_train_val_test(
            augment_train=True
        )
    except TypeError:
        # Compatibility with loaders that do not expose this argument.
        loaded = load_train_val_test()

    (
        X_train,
        y_train,
        X_val,
        y_val,
    ) = normalize_loaded_data(
        loaded
    )

    results = []

    for candidate_id in range(
        trials
    ):
        arch = sample_architecture(
            config
        )

        arch_json = {
            key: (
                list(value)
                if isinstance(
                    value,
                    tuple,
                )
                else value
            )
            for key, value
            in arch.items()
        }

        candidate_dir = (
            output_dir
            / f"candidate_{candidate_id:03d}"
        )

        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            candidate_dir
            / "checkpoint.weights.h5"
        )

        config_path = (
            candidate_dir
            / "config.json"
        )

        history_path = (
            candidate_dir
            / "history.json"
        )

        metrics_path = (
            candidate_dir
            / "metrics.json"
        )

        save_json(
            config_path,
            {
                "candidate_id":
                    candidate_id,

                "architecture":
                    arch_json,

                "epochs_per_candidate":
                    epochs,

                "objective":
                    objective,
            },
        )

        save_json(
            output_dir / "progress.json",
            {
                "status":
                    "training_candidate",

                "completed":
                    candidate_id,

                "total":
                    trials,

                "candidate_id":
                    candidate_id,

                "architecture":
                    arch_json,
            },
        )

        record = {
            "id":
                candidate_id,

            "status":
                "running",

            "architecture":
                arch_json,

            "objective":
                objective,

            "candidate_dir":
                str(candidate_dir),

            "config_path":
                str(config_path),

            "checkpoint_path":
                str(checkpoint_path),

            "history_path":
                str(history_path),

            "metrics_path":
                str(metrics_path),

            "checkpoint_available":
                False,
        }

        try:
            model = build_candidate_model(
                build_nas_model,
                arch,
            )

            params = int(
                model.count_params()
            )

            macs = int(
                estimate_macs(
                    model
                )
            )

            memory_mb_fp32 = (
                params
                * 4.0
                / (
                    1024.0
                    * 1024.0
                )
            )

            model = compile_candidate(
                compile_model,
                model,
            )

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(
                        checkpoint_path
                    ),
                    monitor="val_loss",
                    save_best_only=True,
                    save_weights_only=True,
                    mode="min",
                    verbose=0,
                ),

                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                    verbose=0,
                ),
            ]

            history = model.fit(
                X_train,
                y_train,
                validation_data=(
                    X_val,
                    y_val,
                ),
                epochs=epochs,
                batch_size=128,
                callbacks=callbacks,
                shuffle=True,
                verbose=0,
            )

            save_json(
                history_path,
                json_safe(
                    history.history
                ),
            )

            # Evaluate exactly the weights persisted for this candidate.
            if checkpoint_path.exists():
                model.load_weights(
                    str(checkpoint_path)
                )

            y_prob = model.predict(
                X_val,
                batch_size=512,
                verbose=0,
            )

            y_pred = np.argmax(
                y_prob,
                axis=1,
            )

            if np.asarray(y_val).ndim > 1:
                y_true = np.argmax(
                    y_val,
                    axis=1,
                )
            else:
                y_true = np.asarray(
                    y_val,
                    dtype=np.int64,
                )

            macro_f1 = float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            )

            accuracy = float(
                np.mean(
                    y_pred == y_true
                )
            )

            epochs_ran = int(
                len(
                    history.history.get(
                        "loss",
                        []
                    )
                )
            )

            val_losses = history.history.get(
                "val_loss",
                []
            )

            best_val_loss = (
                float(
                    min(
                        val_losses
                    )
                )
                if val_losses
                else None
            )

            checkpoint_available = (
                checkpoint_path.exists()
                and checkpoint_path.stat().st_size > 0
            )

            record.update(
                {
                    "status":
                        "completed",

                    "macro_f1":
                        macro_f1,

                    "accuracy":
                        accuracy,

                    "params":
                        params,

                    "macs":
                        macs,

                    "memory_mb_fp32":
                        float(
                            memory_mb_fp32
                        ),

                    "epochs_ran":
                        epochs_ran,

                    "best_val_loss":
                        best_val_loss,

                    "checkpoint_available":
                        checkpoint_available,

                    "checkpoint_size_bytes":
                        (
                            int(
                                checkpoint_path
                                .stat()
                                .st_size
                            )
                            if checkpoint_available
                            else 0
                        ),
                }
            )

            save_json(
                metrics_path,
                {
                    "candidate_id":
                        candidate_id,

                    "status":
                        "completed",

                    "macro_f1":
                        macro_f1,

                    "accuracy":
                        accuracy,

                    "params":
                        params,

                    "macs":
                        macs,

                    "memory_mb_fp32":
                        float(
                            memory_mb_fp32
                        ),

                    "epochs_ran":
                        epochs_ran,

                    "best_val_loss":
                        best_val_loss,

                    "checkpoint_available":
                        checkpoint_available,

                    "checkpoint_path":
                        str(
                            checkpoint_path
                        ),
                },
            )

        except Exception as exc:
            record.update(
                {
                    "status":
                        "failed_candidate",

                    "error":
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),

                    "traceback":
                        traceback.format_exc(),

                    "checkpoint_available":
                        (
                            checkpoint_path.exists()
                            and checkpoint_path.stat().st_size > 0
                        ),
                }
            )

            save_json(
                candidate_dir
                / "error.json",
                {
                    "candidate_id":
                        candidate_id,

                    "error":
                        record["error"],

                    "traceback":
                        record["traceback"],
                },
            )

        results.append(
            record
        )

        save_json(
            output_dir
            / "candidates.json",
            results,
        )

        tf.keras.backend.clear_session()

    best = select_best(
        results,
        objective,
    )

    if best is None:
        save_json(
            output_dir
            / "progress.json",
            {
                "status":
                    "failed",

                "completed":
                    trials,

                "total":
                    trials,

                "message":
                    (
                        "Không candidate nào train/evaluate thành công. "
                        "Mở candidates.json để xem error/traceback."
                    ),
            },
        )

        raise RuntimeError(
            "Không candidate NAS nào hoàn thành thành công."
        )

    best_id = best["id"]

    for item in results:
        item["is_best"] = (
            item["id"]
            == best_id
        )

    save_json(
        output_dir
        / "best_candidate.json",
        best,
    )

    save_json(
        output_dir
        / "candidates.json",
        results,
    )

    save_json(
        output_dir
        / "progress.json",
        {
            "status":
                "completed",

            "completed":
                trials,

            "total":
                trials,

            "best_candidate_id":
                best_id,

            "selection_reason":
                best[
                    "selection_reason"
                ],
        },
    )


if __name__ == "__main__":
    main()
