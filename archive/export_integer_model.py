import json
from pathlib import Path

import tensorflow as tf

from config import (
    ARTIFACTS_DIR
)

from quantization.qat_layers import (
    QATWeightWrapper,
    QATActivationWrapper
)

from quantization.integer_export import (
    export_integer_parameters
)


def main():
    model_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    precision_path = (
        ARTIFACTS_DIR
        / "best_layer_precision_config.json"
    )

    scale_path = (
        ARTIFACTS_DIR
        / "activation_scales.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    if not precision_path.exists():
        raise FileNotFoundError(
            precision_path
        )

    if not scale_path.exists():
        raise FileNotFoundError(
            "Chạy calibration trước."
        )

    with Path(
        precision_path
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

    with Path(
        scale_path
    ).open(
        "r",
        encoding="utf-8"
    ) as f:
        scale_data = (
            json.load(f)
        )

    activation_scales = (
        scale_data[
            "activation_scales"
        ]
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

    export_dir = (
        ARTIFACTS_DIR
        / "integer_model"
    )

    metadata = (
        export_integer_parameters(
            model,
            precision_config,
            activation_scales,
            export_dir
        )
    )

    metadata_path = (
        export_dir
        / "model_metadata.json"
    )

    with Path(
        metadata_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        "\n=============================="
    )

    print(
        "INTEGER EXPORT FINISHED"
    )

    print(
        "=============================="
    )

    print(
        "Export directory:"
    )

    print(
        export_dir
    )

    print(
        "\nMetadata:"
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()