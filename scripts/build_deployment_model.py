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

from quantization.deployment_export import (
    export_deployment_model
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
        / "deployment_scales.json"
    )

    required_files = [
        model_path,
        precision_path,
        scale_path
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy:\n{path}"
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

    export_dir = (
        ARTIFACTS_DIR
        / "deployment_model"
    )

    print(
        "\nBuilding deployment model..."
    )

    manifest = (
        export_deployment_model(
            model,
            precision_config,
            scale_data,
            export_dir
        )
    )

    manifest_path = (
        export_dir
        / "deployment_manifest.json"
    )

    with Path(
        manifest_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\n=============================="
    )

    print(
        "DEPLOYMENT EXPORT FINISHED"
    )

    print(
        "=============================="
    )

    print(
        "\nExport directory:"
    )

    print(
        export_dir
    )

    print(
        "\nManifest:"
    )

    print(
        manifest_path
    )

    print(
        "\nLayers exported:",
        len(
            manifest[
                "layers"
            ]
        )
    )

    print(
        "\nINT4 layers:"
    )

    int4_count = 0

    for name, info in (
        manifest[
            "layers"
        ].items()
    ):
        if (
            info["bits"]
            == 4
        ):
            print(
                "INT4  ",
                name
            )

            int4_count += 1

    print(
        "\nINT4 count:",
        int4_count
    )

    print(
        "INT8 count:",
        (
            len(
                manifest[
                    "layers"
                ]
            )
            - int4_count
        )
    )


if __name__ == "__main__":
    main()