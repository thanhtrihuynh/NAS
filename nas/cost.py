import tensorflow as tf


def _dim(v):
    if v is None:
        raise ValueError("Tensor shape chưa xác định.")
    return int(v)


def estimate_macs(model):
    total = 0

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            cin = _dim(layer.input.shape[-1])
            lout = _dim(layer.output.shape[1])
            cout = int(layer.filters)
            k = int(layer.kernel_size[0])

            total += lout * k * cin * cout

        elif isinstance(layer, tf.keras.layers.SeparableConv1D):
            cin = _dim(layer.input.shape[-1])
            lout = _dim(layer.output.shape[1])
            cout = int(layer.filters)
            k = int(layer.kernel_size[0])
            dm = int(layer.depth_multiplier)

            total += lout * k * cin * dm
            total += lout * cin * dm * cout

        elif isinstance(layer, tf.keras.layers.Dense):
            total += _dim(layer.input.shape[-1]) * int(layer.units)

    return int(total)
