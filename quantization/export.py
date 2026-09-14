from pathlib import Path

import numpy as np
import tensorflow as tf

from quantization.fake_quant import quantize_to_integer


def export_integer_weights(model, bits, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {}

    for layer in model.layers:
        if not isinstance(
            layer,
            (
                tf.keras.layers.Conv1D,
                tf.keras.layers.SeparableConv1D,
                tf.keras.layers.Dense,
            ),
        ):
            continue

        weights = layer.get_weights()
        if len(weights) == 0:
            continue

        if isinstance(layer, tf.keras.layers.SeparableConv1D):
            weight_names = ["depthwise_kernel", "pointwise_kernel"]
            if layer.use_bias:
                weight_names.append("bias")
        else:
            weight_names = ["kernel"]
            if getattr(layer, "use_bias", False):
                weight_names.append("bias")

        for weight_name, weight in zip(weight_names, weights):
            q_weight, scale = quantize_to_integer(weight, bits)
            key = f"{layer.name}_{weight_name}"

            export_data[key] = q_weight
            export_data[f"{key}_scale"] = np.asarray(scale, dtype=np.float32)

    np.savez_compressed(output_path, **export_data)
    print("Saved:", output_path)
