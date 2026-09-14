import numpy as np
import tensorflow as tf

from quantization.qat_layers import (
    QATActivationWrapper
)


def get_calibration_layers(
    model
):
    layers = []

    for layer in model.layers:
        if isinstance(
            layer,
            QATActivationWrapper
        ):
            layers.append(
                layer
            )

    return layers


def collect_activation_ranges(
    model,
    X_calibration,
    batch_size=256
):
    calibration_layers = (
        get_calibration_layers(
            model
        )
    )

    if len(
        calibration_layers
    ) == 0:
        raise RuntimeError(
            "Không tìm thấy "
            "QATActivationWrapper."
        )

    outputs = [
        layer.output
        for layer
        in calibration_layers
    ]

    probe_model = (
        tf.keras.Model(
            inputs=model.input,
            outputs=outputs
        )
    )

    max_abs = {
        layer.layer.name: 0.0
        for layer
        in calibration_layers
    }

    num_samples = len(
        X_calibration
    )

    for start in range(
        0,
        num_samples,
        batch_size
    ):
        end = min(
            start + batch_size,
            num_samples
        )

        batch = (
            X_calibration[
                start:end
            ]
        )

        values = probe_model(
            batch,
            training=False
        )

        if not isinstance(
            values,
            (list, tuple)
        ):
            values = [
                values
            ]

        for layer, tensor in zip(
            calibration_layers,
            values
        ):
            name = (
                layer.layer.name
            )

            current = float(
                np.max(
                    np.abs(
                        tensor.numpy()
                    )
                )
            )

            if current > (
                max_abs[name]
            ):
                max_abs[name] = (
                    current
                )

    return max_abs


def ranges_to_scales(
    model,
    max_abs
):
    result = {}

    for layer in (
        get_calibration_layers(
            model
        )
    ):
        name = (
            layer.layer.name
        )

        bits = int(
            layer.bits
        )

        if bits == 8:
            qmax = 127.0
        elif bits == 4:
            qmax = 7.0
        else:
            raise ValueError(
                f"bits={bits}"
            )

        maximum = float(
            max_abs[name]
        )

        scale = (
            maximum / qmax
            if maximum > 1e-12
            else 1.0
        )

        result[name] = {
            "bits": bits,
            "max_abs": maximum,
            "scale": float(
                scale
            )
        }

    return result