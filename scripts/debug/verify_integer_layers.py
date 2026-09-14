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

from integer_reference.ops import (
    quantize_tensor
)

from integer_reference.layer_runner import (
    IntegerLayerRunner
)


VERIFY_SAMPLES = 32


def original_name(
    layer
):
    inner = getattr(
        layer,
        "layer",
        layer
    )

    return inner.name


def build_layer_map(
    model
):
    result = {}

    for layer in model.layers:
        result[
            original_name(layer)
        ] = layer

    return result


def relative_rmse(
    reference,
    test
):
    reference = np.asarray(
        reference,
        dtype=np.float64
    )

    test = np.asarray(
        test,
        dtype=np.float64
    )

    rmse = np.sqrt(
        np.mean(
            (
                reference
                - test
            ) ** 2
        )
    )

    reference_rms = np.sqrt(
        np.mean(
            reference ** 2
        )
    )

    return float(
        rmse
        / (
            reference_rms
            + 1e-12
        )
    )


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

    deployment_dir = (
        ARTIFACTS_DIR
        / "deployment_model"
    )

    manifest_path = (
        deployment_dir
        / "deployment_manifest.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            manifest_path
        )

    with Path(
        manifest_path
    ).open(
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(
            f
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

    layer_map = build_layer_map(
        model
    )

    print(
        "Loading validation data..."
    )

    X_val, y_val, _ = (
        build_xy(
            VAL_CSV,
            RECORDS_DIR
        )
    )

    X_val, _ = (
        to_keras(
            X_val,
            y_val
        )
    )

    rng = (
        np.random.default_rng(
            SEED
        )
    )

    sample_count = min(
        VERIFY_SAMPLES,
        len(X_val)
    )

    indices = rng.choice(
        len(X_val),
        size=sample_count,
        replace=False
    )

    X = X_val[
        indices
    ]

    runner = (
        IntegerLayerRunner(
            deployment_dir,
            manifest
        )
    )

    results = []

    print(
        "\n================================"
    )

    print(
        "INTEGER LAYER VERIFICATION"
    )

    print(
        "================================"
    )

    for (
        layer_name,
        info
    ) in (
        manifest[
            "layers"
        ].items()
    ):
        print(
            "\n--------------------------------"
        )

        print(
            "Layer:",
            layer_name
        )

        if layer_name not in (
            layer_map
        ):
            print(
                "SKIP - không tìm thấy "
                "layer trong QAT model."
            )

            results.append({
                "layer":
                    layer_name,

                "status":
                    "LAYER_NOT_FOUND"
            })

            continue

        wrapped_layer = (
            layer_map[
                layer_name
            ]
        )

        bn_name = (
            info.get(
                "bn_name"
            )
        )

        if (
            bn_name
            and bn_name
            in layer_map
        ):
            target_layer = (
                layer_map[
                    bn_name
                ]
            )
        else:
            target_layer = (
                wrapped_layer
            )

        probe_model = (
            tf.keras.Model(
                inputs=model.input,
                outputs=[
                    wrapped_layer.input,
                    target_layer.output
                ]
            )
        )

        (
            input_float,
            reference_float
        ) = probe_model(
            X,
            training=False
        )

        input_float = (
            input_float.numpy()
        )

        reference_float = (
            reference_float.numpy()
        )

        input_source = (
            info[
                "input_source"
            ]
        )

        tensor_scales = (
            manifest[
                "tensor_scales"
            ]
        )

        if input_source not in (
            tensor_scales
        ):
            print(
                "SKIP - thiếu input scale:",
                input_source
            )

            results.append({
                "layer":
                    layer_name,

                "status":
                    "INPUT_SCALE_MISSING",

                "input_source":
                    input_source
            })

            continue

        input_scale_info = (
            tensor_scales[
                input_source
            ]
        )

        input_scale = float(
            info[
                "input_scale"
            ]
        )

        input_bits = int(
            input_scale_info[
                "bits"
            ]
        )

        x_q = quantize_tensor(
            input_float,
            input_scale,
            input_bits
        )

        try:
            (
                accumulator,
                accumulator_scale
            ) = (
                runner.run_accumulator(
                    layer_name,
                    x_q
                )
            )

        except Exception as exc:
            print(
                "ERROR:",
                exc
            )

            results.append({
                "layer":
                    layer_name,

                "status":
                    "ERROR",

                "error":
                    str(exc)
            })

            continue

        integer_float = (
            accumulator.astype(
                np.float32
            )
            * float(
                accumulator_scale
            )
        )

        if (
            integer_float.shape
            != reference_float.shape
        ):
            print(
                "SHAPE_MISMATCH"
            )

            print(
                "Integer shape:",
                integer_float.shape
            )

            print(
                "Reference shape:",
                reference_float.shape
            )

            results.append({
                "layer":
                    layer_name,

                "status":
                    "SHAPE_MISMATCH",

                "integer_shape":
                    list(
                        integer_float.shape
                    ),

                "reference_shape":
                    list(
                        reference_float.shape
                    )
            })

            continue

        absolute_error = np.abs(
            integer_float
            - reference_float
        )

        mae = float(
            np.mean(
                absolute_error
            )
        )

        max_error = float(
            np.max(
                absolute_error
            )
        )

        nrmse = relative_rmse(
            reference_float,
            integer_float
        )

        print(
            "Type:",
            info["type"]
        )

        print(
            "Bits:",
            info["bits"]
        )

        print(
            "Input source:",
            input_source
        )

        print(
            "Input scale:",
            input_scale
        )

        print(
            "Accumulator scale:",
            accumulator_scale
        )

        print(
            "MAE:",
            mae
        )

        print(
            "Max error:",
            max_error
        )

        print(
            "Relative RMSE:",
            nrmse
        )

        results.append({
            "layer":
                layer_name,

            "status":
                "OK",

            "type":
                info["type"],

            "bits":
                int(
                    info["bits"]
                ),

            "input_source":
                input_source,

            "mae":
                mae,

            "max_error":
                max_error,

            "relative_rmse":
                nrmse
        })

    output_path = (
        ARTIFACTS_DIR
        / "integer_layer_verification.json"
    )

    with Path(
        output_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    error_count = sum(
        1
        for item
        in results
        if item[
            "status"
        ] != "OK"
    )

    print(
        "\n================================"
    )

    print(
        "VERIFICATION FINISHED"
    )

    print(
        "================================"
    )

    print(
        "Total:",
        len(results)
    )

    print(
        "Non-OK:",
        error_count
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()