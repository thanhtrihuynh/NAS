import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    VAL_CSV,
    RECORDS_DIR,
    ARTIFACTS_DIR,
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

from quantization.deployment_calibration import (
    collect_graph_ranges,
    collect_sepconv_internal_ranges
)


CALIBRATION_SAMPLES = 4096
CALIBRATION_BATCH_SIZE = 128


def main():
    np.random.seed(
        SEED
    )

    tf.random.set_seed(
        SEED
    )

    model_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    config_path = (
        ARTIFACTS_DIR
        / "best_layer_precision_config.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model:\n"
            f"{model_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy precision config:\n"
            f"{config_path}"
        )

    with Path(
        config_path
    ).open(
        "r",
        encoding="utf-8"
    ) as f:
        precision_data = (
            json.load(f)
        )

    precision_config = (
        precision_data[
            "precision"
        ]
    )

    print(
        "Loading QAT model..."
    )

    model = (
        tf.keras.models.load_model(
            str(
                model_path
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
        "Model loaded."
    )

    print(
        "\nLoading validation data..."
    )

    X_val, y_val, _ = (
        build_xy(
            VAL_CSV,
            RECORDS_DIR
        )
    )

    X_val_keras, _ = (
        to_keras(
            X_val,
            y_val
        )
    )

    print(
        "Validation shape:",
        X_val_keras.shape
    )

    rng = (
        np.random.default_rng(
            SEED
        )
    )

    sample_count = min(
        CALIBRATION_SAMPLES,
        len(
            X_val_keras
        )
    )

    indices = rng.choice(
        len(
            X_val_keras
        ),
        size=sample_count,
        replace=False
    )

    X_calibration = (
        X_val_keras[
            indices
        ]
    )

    print(
        "Calibration samples:",
        len(
            X_calibration
        )
    )

    print(
        "\nCollecting graph ranges..."
    )

    tensor_scales = (
        collect_graph_ranges(
            model,
            X_calibration,
            precision_config,
            batch_size=
                CALIBRATION_BATCH_SIZE
        )
    )

    print(
        "\nCollecting SeparableConv "
        "internal ranges..."
    )

    sepconv_internal = (
        collect_sepconv_internal_ranges(
            model,
            X_calibration,
            precision_config,
            batch_size=
                CALIBRATION_BATCH_SIZE
        )
    )

    output = {
        "method":
            "symmetric_max_abs",

        "num_calibration_samples":
            int(
                sample_count
            ),

        "input_bits":
            8,

        "tensor_scales":
            tensor_scales,

        "separable_internal_scales":
            sepconv_internal
    }

    output_path = (
        ARTIFACTS_DIR
        / "deployment_scales.json"
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

    print(
        "\n=============================="
    )

    print(
        "DEPLOYMENT CALIBRATION DONE"
    )

    print(
        "=============================="
    )

    print(
        "Tensor scales:",
        len(
            tensor_scales
        )
    )

    print(
        "Separable internal scales:",
        len(
            sepconv_internal
        )
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()