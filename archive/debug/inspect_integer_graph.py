import json
from pathlib import Path

import tensorflow as tf

from config import ARTIFACTS_DIR

from quantization.qat_layers import (
    QATWeightWrapper,
    QATActivationWrapper
)


def clean_shape(shape):
    if shape is None:
        return None

    if isinstance(shape, (list, tuple)):
        try:
            return [
                int(x) if x is not None else None
                for x in shape
            ]
        except Exception:
            return str(shape)

    return str(shape)


def tensor_name(tensor):
    try:
        history = tensor._keras_history

        operation = history.operation

        return operation.name

    except Exception:
        return None


def get_inner_layer(layer):
    return getattr(
        layer,
        "layer",
        layer
    )


def main():
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

    print(
        "Loading QAT model..."
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

    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    records = []

    print()
    print(
        "=" * 100
    )
    print(
        "QAT MODEL GRAPH"
    )
    print(
        "=" * 100
    )

    for index, layer in enumerate(
        model.layers
    ):
        inner = get_inner_layer(
            layer
        )

        wrapper_type = (
            layer.__class__.__name__
        )

        inner_type = (
            inner.__class__.__name__
        )

        original_name = (
            inner.name
        )

        inputs = (
            layer.input
            if isinstance(
                layer.input,
                (list, tuple)
            )
            else [layer.input]
        )

        input_sources = []

        for tensor in inputs:
            input_sources.append(
                tensor_name(
                    tensor
                )
            )

        try:
            output_shape = (
                layer.output.shape
            )

            output_shape = (
                clean_shape(
                    output_shape
                )
            )

        except Exception:
            output_shape = None

        bits = None
        manifest_input = None

        if (
            original_name
            in manifest["layers"]
        ):
            info = (
                manifest[
                    "layers"
                ][original_name]
            )

            bits = info.get(
                "bits"
            )

            manifest_input = (
                info.get(
                    "input_source"
                )
            )

        record = {
            "index":
                index,

            "keras_name":
                layer.name,

            "original_name":
                original_name,

            "wrapper_type":
                wrapper_type,

            "inner_type":
                inner_type,

            "input_sources":
                input_sources,

            "output_shape":
                output_shape,

            "bits":
                bits,

            "manifest_input_source":
                manifest_input
        }

        records.append(
            record
        )

        print(
            f"{index:02d} | "
            f"{original_name:<28} | "
            f"{inner_type:<24} | "
            f"in={str(input_sources):<35} | "
            f"out={output_shape}"
        )

    output_path = (
        ARTIFACTS_DIR
        / "integer_graph.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "=" * 100
    )
    print(
        "IMPORTANT GRAPH NODES"
    )
    print(
        "=" * 100
    )

    keywords = [
        "stem",
        "block1",
        "block2",
        "block3",
        "block4",
        "pool1",
        "pool2",
        "pool3",
        "global",
        "head",
        "class"
    ]

    for record in records:
        name = (
            record[
                "original_name"
            ].lower()
        )

        if any(
            keyword in name
            for keyword in keywords
        ):
            print(
                f"{record['original_name']:<30} "
                f"{record['inner_type']:<25} "
                f"<- {record['input_sources']}"
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