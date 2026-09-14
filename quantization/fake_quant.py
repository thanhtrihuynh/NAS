import numpy as np
import tensorflow as tf


def quant_limits(bits):
    if bits == 8:
        return -127, 127
    if bits == 4:
        return -7, 7
    raise ValueError("Only INT4 and INT8 are supported.")


def fake_quant_tensor(x, bits):
    x = tf.cast(x, tf.float32)
    qmin, qmax = quant_limits(bits)

    max_abs = tf.reduce_max(tf.abs(x))
    scale = tf.maximum(max_abs / float(qmax), 1e-8)

    q = tf.round(x / scale)
    q = tf.clip_by_value(q, float(qmin), float(qmax))
    return q * scale


def fake_quant_numpy(x, bits):
    x = np.asarray(x, dtype=np.float32)
    qmin, qmax = quant_limits(bits)

    max_abs = float(np.max(np.abs(x)))
    if max_abs < 1e-8:
        return x.copy()

    scale = max_abs / float(qmax)
    q = np.round(x / scale)
    q = np.clip(q, qmin, qmax)
    return (q * scale).astype(np.float32)


def quantize_to_integer(x, bits):
    x = np.asarray(x, dtype=np.float32)
    qmin, qmax = quant_limits(bits)

    max_abs = float(np.max(np.abs(x)))

    if max_abs < 1e-8:
        scale = np.float32(1.0)
        q = np.zeros_like(x, dtype=np.int8)
        return q, scale

    scale = np.float32(max_abs / float(qmax))
    q = np.round(x / scale)
    q = np.clip(q, qmin, qmax).astype(np.int8)
    return q, scale


@tf.keras.utils.register_keras_serializable(package="ECGNAS")
class FakeQuantOutput(tf.keras.layers.Wrapper):
    def __init__(self, layer, bits, quantize_output=True, **kwargs):
        super().__init__(layer, **kwargs)
        self.bits = int(bits)
        self.quantize_output = bool(quantize_output)

    def call(self, inputs, training=None, mask=None, **kwargs):
        try:
            output = self.layer(inputs, training=training, **kwargs)
        except TypeError:
            output = self.layer(inputs, **kwargs)

        if not self.quantize_output:
            return output

        return tf.nest.map_structure(
            lambda x: fake_quant_tensor(x, self.bits),
            output,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bits": self.bits,
                "quantize_output": self.quantize_output,
            }
        )
        return config


def should_quantize_output(layer):
    quantized_types = (
        tf.keras.layers.Activation,
        tf.keras.layers.MaxPooling1D,
        tf.keras.layers.Add,
        tf.keras.layers.Concatenate,
        tf.keras.layers.GlobalAveragePooling1D,
        tf.keras.layers.GlobalMaxPooling1D,
    )

    if layer.name == "class_output":
        return False

    return isinstance(layer, quantized_types)


WEIGHT_LAYER_TYPES = (
    tf.keras.layers.Conv1D,
    tf.keras.layers.SeparableConv1D,
    tf.keras.layers.Dense,
)


def build_uniform_quant_model(base_model, bits):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")

    def clone_function(layer):
        cloned_layer = layer.__class__.from_config(layer.get_config())
        quantize_activation = should_quantize_output(layer)

        return FakeQuantOutput(
            cloned_layer,
            bits=bits,
            quantize_output=quantize_activation,
            name=f"q{bits}_{layer.name}",
        )

    quant_model = tf.keras.models.clone_model(
        base_model,
        clone_function=clone_function,
    )

    target_layers = {}
    for layer in quant_model.layers:
        if isinstance(layer, FakeQuantOutput):
            target_layers[layer.layer.name] = layer.layer

    for original_layer in base_model.layers:
        if original_layer.name not in target_layers:
            continue

        target_layer = target_layers[original_layer.name]
        original_weights = original_layer.get_weights()

        if len(original_weights) == 0:
            continue

        if isinstance(original_layer, WEIGHT_LAYER_TYPES):
            new_weights = [fake_quant_numpy(w, bits) for w in original_weights]
            target_layer.set_weights(new_weights)
        else:
            target_layer.set_weights(original_weights)

    quant_model.trainable = False
    return quant_model
