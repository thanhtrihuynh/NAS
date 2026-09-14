import argparse
import json
import shutil
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

from quantization.fixed_point import (
    quantize_symmetric
)

from quantization.deployment_export import (
    activation_name_for_layer,
    bn_name_for_layer
)


CALIBRATION_SAMPLES = 4096
BATCH_SIZE = 128

MAX_VALUES_PER_BATCH = 25000


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9
    )

    return parser.parse_args()


def original_name(layer):
    inner = getattr(
        layer,
        "layer",
        layer
    )

    return inner.name


def build_layer_map(model):
    result = {}

    for layer in model.layers:
        result[
            original_name(layer)
        ] = layer

    return result


def sample_absolute_values(
    values,
    rng
):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = np.abs(
        values.reshape(-1)
    )

    if (
        len(values)
        <= MAX_VALUES_PER_BATCH
    ):
        return values

    indices = rng.choice(
        len(values),
        size=MAX_VALUES_PER_BATCH,
        replace=False
    )

    return values[
        indices
    ]


def compute_percentile_info(
    sampled_values,
    true_max,
    percentile,
    bits
):
    if not sampled_values:
        raise RuntimeError(
            "Không có calibration samples."
        )

    values = np.concatenate(
        sampled_values
    )

    clip_abs = float(
        np.percentile(
            values,
            percentile
        )
    )

    qmax = (
        127.0
        if bits == 8
        else 7.0
    )

    if clip_abs < 1e-12:
        scale = 1.0

    else:
        scale = (
            clip_abs
            / qmax
        )

    return {
        "bits":
            int(bits),

        "max_abs":
            float(true_max),

        "clip_abs":
            clip_abs,

        "scale":
            float(scale),

        "calibration":
            "percentile",

        "percentile":
            float(percentile)
    }


def collect_activation_percentiles(
    model,
    X_calibration,
    int4_layers,
    percentile,
    rng
):
    layer_map = build_layer_map(
        model
    )

    targets = []

    for layer_name in int4_layers:
        activation_name = (
            activation_name_for_layer(
                layer_name
            )
        )

        bn_name = (
            bn_name_for_layer(
                layer_name
            )
        )

        if activation_name is None:
            print(
                "Skip activation:",
                layer_name
            )

            continue

        if bn_name is None:
            raise RuntimeError(
                "Không xác định được BN "
                f"cho {layer_name}"
            )

        if bn_name not in layer_map:
            raise KeyError(
                "Không tìm thấy QAT layer: "
                f"{bn_name}"
            )

        targets.append({
            "weight_layer":
                layer_name,

            "bn_name":
                bn_name,

            "activation_name":
                activation_name
        })

    probe_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            layer_map[
                item["bn_name"]
            ].output
            for item in targets
        ]
    )

    samples = {
        item[
            "activation_name"
        ]: []
        for item in targets
    }

    true_max = {
        item[
            "activation_name"
        ]: 0.0
        for item in targets
    }

    total = len(
        X_calibration
    )

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            total
        )

        print(
            f"\rActivation calibration: "
            f"{end}/{total}",
            end=""
        )

        batch = (
            X_calibration[
                start:end
            ]
        )

        outputs = probe_model(
            batch,
            training=False
        )

        if not isinstance(
            outputs,
            (list, tuple)
        ):
            outputs = [
                outputs
            ]

        for item, bn_output in zip(
            targets,
            outputs
        ):
            # Model architecture:
            # SepConv -> BN -> GELU
            activation = (
                tf.keras.activations.gelu(
                    bn_output,
                    approximate=False
                )
            )

            activation = (
                activation.numpy()
            )

            name = (
                item[
                    "activation_name"
                ]
            )

            current_max = float(
                np.max(
                    np.abs(
                        activation
                    )
                )
            )

            true_max[name] = max(
                true_max[name],
                current_max
            )

            sampled = (
                sample_absolute_values(
                    activation,
                    rng
                )
            )

            samples[
                name
            ].append(
                sampled
            )

    print()

    result = {}

    for item in targets:
        name = (
            item[
                "activation_name"
            ]
        )

        result[name] = (
            compute_percentile_info(
                samples[name],
                true_max[name],
                percentile,
                bits=4
            )
        )

    return result


def collect_depthwise_percentiles(
    model,
    X_calibration,
    int4_layers,
    percentile,
    rng
):
    wrappers = {}

    for layer in model.layers:
        if (
            isinstance(
                layer,
                QATWeightWrapper
            )
            and isinstance(
                layer.layer,
                tf.keras.layers.SeparableConv1D
            )
        ):
            wrappers[
                layer.layer.name
            ] = layer

    targets = []

    for layer_name in int4_layers:
        if layer_name not in wrappers:
            raise KeyError(
                "Không tìm thấy "
                "SeparableConv wrapper: "
                f"{layer_name}"
            )

        targets.append(
            wrappers[
                layer_name
            ]
        )

    probe_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            wrapper.input
            for wrapper in targets
        ]
    )

    kernels = {}

    for wrapper in targets:
        layer = wrapper.layer

        (
            q_kernel,
            kernel_scale
        ) = quantize_symmetric(
            layer.depthwise_kernel.numpy(),
            4
        )

        dequantized_kernel = (
            q_kernel.astype(
                np.float32
            )
            * float(
                kernel_scale
            )
        )

        kernels[
            layer.name
        ] = tf.convert_to_tensor(
            dequantized_kernel,
            dtype=tf.float32
        )

    samples = {
        wrapper.layer.name: []
        for wrapper in targets
    }

    true_max = {
        wrapper.layer.name: 0.0
        for wrapper in targets
    }

    total = len(
        X_calibration
    )

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            total
        )

        print(
            f"\rDepthwise calibration: "
            f"{end}/{total}",
            end=""
        )

        batch = (
            X_calibration[
                start:end
            ]
        )

        inputs = probe_model(
            batch,
            training=False
        )

        if not isinstance(
            inputs,
            (list, tuple)
        ):
            inputs = [
                inputs
            ]

        for wrapper, x in zip(
            targets,
            inputs
        ):
            layer = (
                wrapper.layer
            )

            kernel = kernels[
                layer.name
            ]

            y = tf.keras.ops.depthwise_conv(
                tf.cast(
                    x,
                    tf.float32
                ),
                kernel,
                strides=int(
                    layer.strides[0]
                ),
                padding=
                    layer.padding,
                data_format=
                    (
                        layer.data_format
                        or "channels_last"
                    ),
                dilation_rate=int(
                    layer.dilation_rate[0]
                )
            )

            y = y.numpy()

            current_max = float(
                np.max(
                    np.abs(y)
                )
            )

            true_max[
                layer.name
            ] = max(
                true_max[
                    layer.name
                ],
                current_max
            )

            samples[
                layer.name
            ].append(
                sample_absolute_values(
                    y,
                    rng
                )
            )

    print()

    result = {}

    for wrapper in targets:
        name = (
            wrapper.layer.name
        )

        result[name] = (
            compute_percentile_info(
                samples[name],
                true_max[name],
                percentile,
                bits=4
            )
        )

    return result


def main():
    args = parse_args()

    percentile = float(
        args.percentile
    )

    if not (
        90.0
        < percentile
        <= 100.0
    ):
        raise ValueError(
            "Percentile phải nằm "
            "trong khoảng (90, 100]."
        )

    np.random.seed(
        SEED
    )

    tf.random.set_seed(
        SEED
    )

    rng = np.random.default_rng(
        SEED
    )

    model_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    precision_path = (
        ARTIFACTS_DIR
        / "best_layer_precision_config.json"
    )

    current_scale_path = (
        ARTIFACTS_DIR
        / "deployment_scales.json"
    )

    baseline_scale_path = (
        ARTIFACTS_DIR
        / "deployment_scales_maxabs.json"
    )

    # Luôn lấy max-abs baseline làm gốc,
    # không cộng dồn các lần percentile.
    if not baseline_scale_path.exists():
        shutil.copy2(
            current_scale_path,
            baseline_scale_path
        )

    with baseline_scale_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        scale_data = json.load(f)

    with precision_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        precision_data = json.load(f)

    precision_config = (
        precision_data[
            "precision"
        ]
    )

    int4_layers = [
        name
        for name, bits
        in precision_config.items()
        if int(bits) == 4
    ]

    print()
    print("=" * 70)
    print("INT4 PERCENTILE RECALIBRATION")
    print("=" * 70)

    print(
        "Percentile:",
        percentile
    )

    print(
        "INT4 layers:"
    )

    for name in int4_layers:
        print(
            " -",
            name
        )

    print(
        "\nLoading QAT model..."
    )

    model = tf.keras.models.load_model(
        str(model_path),
        custom_objects={
            "QATWeightWrapper":
                QATWeightWrapper,

            "QATActivationWrapper":
                QATActivationWrapper
        },
        compile=False
    )

    print(
        "Loading validation data..."
    )

    X_val, y_val, _ = build_xy(
        VAL_CSV,
        RECORDS_DIR
    )

    X_val, _ = to_keras(
        X_val,
        y_val
    )

    sample_count = min(
        CALIBRATION_SAMPLES,
        len(X_val)
    )

    indices = rng.choice(
        len(X_val),
        size=sample_count,
        replace=False
    )

    X_calibration = (
        X_val[
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
        "\nCollecting INT4 "
        "activation percentiles..."
    )

    activation_info = (
        collect_activation_percentiles(
            model,
            X_calibration,
            int4_layers,
            percentile,
            rng
        )
    )

    print(
        "\nCollecting INT4 "
        "Depthwise percentiles..."
    )

    depthwise_info = (
        collect_depthwise_percentiles(
            model,
            X_calibration,
            int4_layers,
            percentile,
            rng
        )
    )

    print()
    print("=" * 70)
    print("SCALE CHANGES")
    print("=" * 70)

    for (
        activation_name,
        new_info
    ) in activation_info.items():
        old_info = (
            scale_data[
                "tensor_scales"
            ][
                activation_name
            ]
        )

        old_scale = float(
            old_info[
                "scale"
            ]
        )

        new_scale = float(
            new_info[
                "scale"
            ]
        )

        print(
            f"{activation_name:<24} "
            f"{old_scale:.8f} "
            f"-> {new_scale:.8f} "
            f"ratio="
            f"{new_scale / old_scale:.4f}"
        )

        scale_data[
            "tensor_scales"
        ][
            activation_name
        ] = new_info

    print()

    for (
        layer_name,
        new_info
    ) in depthwise_info.items():
        old_info = (
            scale_data[
                "separable_internal_scales"
            ][
                layer_name
            ]
        )

        old_scale = float(
            old_info[
                "scale"
            ]
        )

        new_scale = float(
            new_info[
                "scale"
            ]
        )

        print(
            f"{layer_name:<24} "
            f"DW {old_scale:.8f} "
            f"-> {new_scale:.8f} "
            f"ratio="
            f"{new_scale / old_scale:.4f}"
        )

        scale_data[
            "separable_internal_scales"
        ][
            layer_name
        ] = new_info

    scale_data[
        "int4_percentile_calibration"
    ] = {
        "enabled":
            True,

        "percentile":
            percentile,

        "samples":
            int(
                sample_count
            )
    }

    with current_scale_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            scale_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    suffix = (
        str(percentile)
        .replace(
            ".",
            "_"
        )
    )

    variant_path = (
        ARTIFACTS_DIR
        / (
            "deployment_scales_"
            f"p{suffix}.json"
        )
    )

    shutil.copy2(
        current_scale_path,
        variant_path
    )

    print()
    print("=" * 70)
    print("RECALIBRATION DONE")
    print("=" * 70)

    print(
        "Active scale file:"
    )

    print(
        current_scale_path
    )

    print(
        "\nVariant copy:"
    )

    print(
        variant_path
    )


if __name__ == "__main__":
    main()