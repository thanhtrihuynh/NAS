import argparse
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

from quantization.fixed_point import (
    qrange
)

from integer_reference.model import (
    IntegerReferenceModel
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--samples",
        type=int,
        default=64
    )

    return parser.parse_args()


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
            original_name(
                layer
            )
        ] = layer

    return result


def calculate_metrics(
    reference,
    integer_float
):
    reference = np.asarray(
        reference,
        dtype=np.float64
    )

    integer_float = np.asarray(
        integer_float,
        dtype=np.float64
    )

    error = (
        integer_float
        - reference
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    reference_rms = float(
        np.sqrt(
            np.mean(
                reference ** 2
            )
        )
    )

    relative_rmse = float(
        rmse
        / (
            reference_rms
            + 1e-12
        )
    )

    ref_flat = reference.reshape(
        -1
    )

    int_flat = integer_float.reshape(
        -1
    )

    denominator = (
        np.linalg.norm(
            ref_flat
        )
        * np.linalg.norm(
            int_flat
        )
    )

    if denominator > 0:
        cosine = float(
            np.dot(
                ref_flat,
                int_flat
            )
            / denominator
        )

    else:
        cosine = 1.0

    max_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "relative_rmse":
            relative_rmse,

        "cosine_similarity":
            cosine,

        "max_error":
            max_error
    }


def main():
    args = parse_args()

    np.random.seed(
        SEED
    )

    tf.random.set_seed(
        SEED
    )

    # ==========================================
    # Data
    # ==========================================

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

    rng = np.random.default_rng(
        SEED
    )

    sample_count = min(
        args.samples,
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

    print(
        "Samples:",
        len(X)
    )

    # ==========================================
    # QAT
    # ==========================================

    qat_path = (
        ARTIFACTS_DIR
        / "best_mixed_precision_qat.keras"
    )

    print(
        "Loading QAT model..."
    )

    qat_model = (
        tf.keras.models.load_model(
            str(qat_path),
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
        qat_model
    )

    # ==========================================
    # Integer
    # ==========================================

    deployment_dir = (
        ARTIFACTS_DIR
        / "deployment_model"
    )

    integer_model = (
        IntegerReferenceModel(
            deployment_dir
        )
    )

    print(
        "Running Integer Reference..."
    )

    integer_result = (
        integer_model.forward(
            X,
            return_trace=True
        )
    )

    trace = integer_result[
        "trace"
    ]

    manifest = (
        integer_model.manifest
    )

    # Chỉ so tensor có semantic name
    # tồn tại thật trong QAT graph.
    compare_names = [
        name
        for name in trace
        if name in layer_map
    ]

    print()
    print(
        "Nodes compared:"
    )

    for name in compare_names:
        print(
            " -",
            name
        )

    if not compare_names:
        raise RuntimeError(
            "Không tìm thấy node chung."
        )

    # ==========================================
    # Probe QAT
    # ==========================================

    probe_model = tf.keras.Model(
        inputs=qat_model.input,
        outputs=[
            layer_map[
                name
            ].output
            for name
            in compare_names
        ]
    )

    qat_values = probe_model(
        X,
        training=False
    )

    if not isinstance(
        qat_values,
        (list, tuple)
    ):
        qat_values = [
            qat_values
        ]

    records = []

    print()
    print(
        "=" * 120
    )

    print(
        "SEMANTICALLY ALIGNED "
        "QAT vs INTEGER TRACE"
    )

    print(
        "=" * 120
    )

    # ==========================================
    # Individual node metrics
    # ==========================================

    for (
        name,
        qat_tensor
    ) in zip(
        compare_names,
        qat_values
    ):
        reference = (
            qat_tensor.numpy()
        )

        integer_tensor = (
            trace[
                name
            ][
                "tensor"
            ]
        )

        scale = float(
            trace[
                name
            ][
                "scale"
            ]
        )

        integer_float = (
            integer_tensor.astype(
                np.float32
            )
            * scale
        )

        if (
            reference.shape
            != integer_float.shape
        ):
            print(
                f"{name:<28} "
                "SHAPE MISMATCH "
                f"{reference.shape} "
                "vs "
                f"{integer_float.shape}"
            )

            continue

        metrics = calculate_metrics(
            reference,
            integer_float
        )

        bits = int(
            manifest[
                "tensor_scales"
            ].get(
                name,
                {}
            ).get(
                "bits",
                8
            )
        )

        qmin, qmax = qrange(
            bits
        )

        saturation_rate = float(
            np.mean(
                (
                    integer_tensor
                    <= qmin
                )
                |
                (
                    integer_tensor
                    >= qmax
                )
            )
        )

        record = {
            "name":
                name,

            "shape":
                list(
                    reference.shape
                ),

            "scale":
                scale,

            "bits":
                bits,

            "saturation_rate":
                saturation_rate,

            **metrics
        }

        records.append(
            record
        )

        print(
            f"{name:<28} "
            f"INT{bits:<2} "
            f"NRMSE="
            f"{metrics['relative_rmse']:.6f} "
            f"cos="
            f"{metrics['cosine_similarity']:.6f} "
            f"sat="
            f"{saturation_rate * 100:.3f}%"
        )

    # ==========================================
    # Lookup table
    # ==========================================

    record_map = {
        item[
            "name"
        ]: item
        for item in records
    }

    # ==========================================
    # Main backbone only
    #
    # Đây mới là nơi hợp lệ để tính
    # sự tăng NRMSE giữa các stage.
    # ==========================================

    main_path = [
        "stem_act",

        "block1_concat",
        "block1_out_bn",
        "block1_add",
        "block1_act",
        "pool1",

        "block2_concat",
        "block2_out_bn",
        "block2_add",
        "block2_act",
        "pool2",

        "block3_concat",
        "block3_out_bn",
        "block3_add",
        "block3_act",
        "pool3",

        "block4_concat",
        "block4_out_bn",
        "block4_add",
        "block4_act",

        "gap",
        "gmp",
        "head_concat",
        "head_act"
    ]

    print()
    print(
        "=" * 120
    )

    print(
        "MAIN PATH ERROR GROWTH"
    )

    print(
        "=" * 120
    )

    previous_nrmse = None

    growth_records = []

    for name in main_path:
        if name not in record_map:
            continue

        item = record_map[
            name
        ]

        current = float(
            item[
                "relative_rmse"
            ]
        )

        if previous_nrmse is None:
            growth = current

        else:
            growth = (
                current
                - previous_nrmse
            )

        previous_nrmse = current

        growth_records.append(
            (
                name,
                growth,
                current
            )
        )

        print(
            f"{name:<28} "
            f"NRMSE="
            f"{current:.6f} "
            f"delta="
            f"{growth:+.6f}"
        )

    # ==========================================
    # Highest absolute errors
    # ==========================================

    print()
    print(
        "=" * 120
    )

    print(
        "HIGHEST NRMSE NODES"
    )

    print(
        "=" * 120
    )

    highest_nrmse = sorted(
        records,
        key=lambda item:
            item[
                "relative_rmse"
            ],
        reverse=True
    )

    for item in highest_nrmse[
        :15
    ]:
        print(
            f"{item['name']:<28} "
            f"NRMSE="
            f"{item['relative_rmse']:.6f} "
            f"cos="
            f"{item['cosine_similarity']:.6f} "
            f"sat="
            f"{item['saturation_rate'] * 100:.3f}%"
        )

    # ==========================================
    # Biggest increases on main path
    # ==========================================

    print()
    print(
        "=" * 120
    )

    print(
        "LARGEST MAIN-PATH ERROR INCREASES"
    )

    print(
        "=" * 120
    )

    growth_records = sorted(
        growth_records,
        key=lambda item:
            item[1],
        reverse=True
    )

    for (
        name,
        growth,
        current
    ) in growth_records[
        :10
    ]:
        print(
            f"{name:<28} "
            f"delta="
            f"{growth:+.6f} "
            f"NRMSE="
            f"{current:.6f}"
        )

    # ==========================================
    # Save
    # ==========================================

    output_path = (
        ARTIFACTS_DIR
        / "integer_trace_debug.json"
    )

    with Path(
        output_path
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "samples":
                    int(
                        sample_count
                    ),

                "records":
                    records,

                "main_path_growth": [
                    {
                        "name":
                            name,

                        "delta":
                            float(
                                growth
                            ),

                        "relative_rmse":
                            float(
                                current
                            )
                    }
                    for (
                        name,
                        growth,
                        current
                    )
                    in growth_records
                ]
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Saved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()