import json
from pathlib import Path

import numpy as np
import tensorflow as tf

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

from quantization.calibration import (
    collect_activation_ranges,
    ranges_to_scales
)


CALIBRATION_SAMPLES = 4096
CALIBRATION_BATCH_SIZE = 256


def main():
    np.random.seed(
        SEED
    )

    model_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    print(
        "Loading final QAT model..."
    )

    model = (
        tf.keras.models.load_model(
            str(model_path),
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
        "Loading calibration data..."
    )

    X_val, y_val, _ = build_xy(
        VAL_CSV,
        RECORDS_DIR
    )

    X_val, _ = to_keras(
        X_val,
        y_val
    )

    rng = (
        np.random.default_rng(
            SEED
        )
    )

    n = min(
        CALIBRATION_SAMPLES,
        len(X_val)
    )

    indices = rng.choice(
        len(X_val),
        size=n,
        replace=False
    )

    X_cal = (
        X_val[
            indices
        ]
    )

    print(
        "Calibration samples:",
        len(X_cal)
    )

    max_abs = (
        collect_activation_ranges(
            model,
            X_cal,
            batch_size=
                CALIBRATION_BATCH_SIZE
        )
    )

    scales = ranges_to_scales(
        model,
        max_abs
    )

    output = {
        "num_calibration_samples":
            int(len(X_cal)),

        "method":
            "symmetric_max_abs",

        "activation_scales":
            scales
    }

    output_path = (
        ARTIFACTS_DIR
        / "activation_scales.json"
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
            indent=2
        )

    print(
        "\n=============================="
    )

    print(
        "CALIBRATION FINISHED"
    )

    print(
        "=============================="
    )

    for name, item in (
        scales.items()
    ):
        print(
            f"{name:<30} "
            f"A{item['bits']} "
            f"scale="
            f"{item['scale']:.10f}"
        )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()