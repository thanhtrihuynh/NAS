import numpy as np
import tensorflow as tf


def estimate_macs(model):
    total_macs = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            cin = int(layer.input.shape[-1])
            cout = int(layer.filters)
            length = int(layer.output.shape[1])
            kernel = int(layer.kernel_size[0])
            total_macs += length * cin * cout * kernel

        elif isinstance(layer, tf.keras.layers.SeparableConv1D):
            cin = int(layer.input.shape[-1])
            cout = int(layer.filters)
            length = int(layer.output.shape[1])
            kernel = int(layer.kernel_size[0])
            depth_multiplier = int(layer.depth_multiplier)

            depthwise = length * cin * kernel * depth_multiplier
            pointwise = length * cin * depth_multiplier * cout
            total_macs += depthwise + pointwise

        elif isinstance(layer, tf.keras.layers.Dense):
            total_macs += int(layer.input.shape[-1]) * int(layer.units)

    return int(total_macs)


def count_weight_parameters(model):
    total = 0
    for layer in model.layers:
        if isinstance(
            layer,
            (
                tf.keras.layers.Conv1D,
                tf.keras.layers.SeparableConv1D,
                tf.keras.layers.Dense,
            ),
        ):
            for weight in layer.get_weights():
                total += int(np.prod(weight.shape))
    return total


def estimate_uniform_weight_memory_bytes(model, bits):
    if bits not in (4, 8, 32):
        raise ValueError("bits phải là 4, 8 hoặc 32")
    return count_weight_parameters(model) * bits / 8.0
