import numpy as np
import tensorflow as tf

from quantization.fake_quant import (
    FakeQuantOutput,
    fake_quant_numpy
)


WEIGHT_LAYER_TYPES = (
    tf.keras.layers.Conv1D,
    tf.keras.layers.SeparableConv1D,
    tf.keras.layers.Dense
)


def get_searchable_layer_names(model):
    names = []

    for layer in model.layers:
        if isinstance(layer, WEIGHT_LAYER_TYPES):
            names.append(layer.name)

    return names


def get_weight_bits(
    layer_name,
    precision_config,
    default_bits=8
):
    bits = int(
        precision_config.get(
            layer_name,
            default_bits
        )
    )

    if bits not in (4, 8):
        raise ValueError(
            f"{layer_name}: bits phải là 4 hoặc 8"
        )

    return bits


def activation_controller(layer_name):
    if layer_name == "stem_act":
        return "stem_conv"

    if layer_name == "head_act":
        return "head_dense"

    mappings = [
        ("_b1_act", "_b1_sepconv"),
        ("_b2_act", "_b2_sepconv"),
        ("_b3_act", "_b3_sepconv"),
        ("_pool_act", "_pool_proj"),
        ("_out_bn", "_out_proj"),
        ("_shortcut_bn", "_shortcut"),
    ]

    for suffix, target_suffix in mappings:
        if layer_name.endswith(suffix):
            prefix = layer_name[
                :-len(suffix)
            ]

            return (
                prefix
                + target_suffix
            )

    return None


def get_activation_bits(
    layer,
    precision_config,
    default_bits=8
):
    controller = activation_controller(
        layer.name
    )

    if controller is not None:
        return get_weight_bits(
            controller,
            precision_config,
            default_bits
        )

    # Residual/block boundaries giữ A8
    if isinstance(
        layer,
        (
            tf.keras.layers.MaxPooling1D,
            tf.keras.layers.Add,
            tf.keras.layers.Concatenate,
            tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalMaxPooling1D
        )
    ):
        return 8

    # Activation cuối mỗi residual block giữ A8
    if (
        isinstance(
            layer,
            tf.keras.layers.Activation
        )
        and layer.name.startswith("block")
        and layer.name.endswith("_act")
    ):
        return 8

    # Softmax cuối giữ FP32
    if layer.name == "class_output":
        return None

    return None


def quantize_weight_layer(
    layer,
    weights,
    bits
):
    if len(weights) == 0:
        return weights

    if isinstance(
        layer,
        tf.keras.layers.SeparableConv1D
    ):
        new_weights = [
            fake_quant_numpy(
                weights[0],
                bits
            ),
            fake_quant_numpy(
                weights[1],
                bits
            )
        ]

        # Bias giữ precision cao
        if len(weights) > 2:
            new_weights.append(
                weights[2]
            )

        return new_weights

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv1D,
            tf.keras.layers.Dense
        )
    ):
        new_weights = [
            fake_quant_numpy(
                weights[0],
                bits
            )
        ]

        # Bias không lượng tử hóa xuống INT4/8
        if len(weights) > 1:
            new_weights.extend(
                weights[1:]
            )

        return new_weights

    return weights


def build_layerwise_mixed_precision_model(
    base_model,
    precision_config,
    default_bits=8
):
    for layer_name, bits in (
        precision_config.items()
    ):
        if int(bits) not in (4, 8):
            raise ValueError(
                f"{layer_name}: "
                "precision phải là INT4 hoặc INT8"
            )

    def clone_function(layer):
        cloned_layer = (
            layer.__class__.from_config(
                layer.get_config()
            )
        )

        activation_bits = (
            get_activation_bits(
                layer,
                precision_config,
                default_bits
            )
        )

        if activation_bits is None:
            return cloned_layer

        return FakeQuantOutput(
            cloned_layer,
            bits=activation_bits,
            quantize_output=True,
            name=f"mixed_{layer.name}"
        )

    mixed_model = (
        tf.keras.models.clone_model(
            base_model,
            clone_function=clone_function
        )
    )

    # Ánh xạ layer gốc -> layer clone
    target_layers = {}

    for layer in mixed_model.layers:
        if isinstance(
            layer,
            FakeQuantOutput
        ):
            target_layers[
                layer.layer.name
            ] = layer.layer
        else:
            target_layers[
                layer.name
            ] = layer

    # Copy/quantize weights
    for original_layer in (
        base_model.layers
    ):
        if (
            original_layer.name
            not in target_layers
        ):
            continue

        target_layer = (
            target_layers[
                original_layer.name
            ]
        )

        original_weights = (
            original_layer.get_weights()
        )

        if len(original_weights) == 0:
            continue

        if isinstance(
            original_layer,
            WEIGHT_LAYER_TYPES
        ):
            bits = get_weight_bits(
                original_layer.name,
                precision_config,
                default_bits
            )

            new_weights = (
                quantize_weight_layer(
                    original_layer,
                    original_weights,
                    bits
                )
            )

            target_layer.set_weights(
                new_weights
            )

        else:
            target_layer.set_weights(
                original_weights
            )

    mixed_model.trainable = False

    return mixed_model


def layer_macs(layer):
    if isinstance(
        layer,
        tf.keras.layers.Conv1D
    ):
        length = int(
            layer.output.shape[1]
        )

        cin = int(
            layer.input.shape[-1]
        )

        cout = int(
            layer.filters
        )

        kernel = int(
            layer.kernel_size[0]
        )

        return (
            length
            * cin
            * cout
            * kernel
        )

    if isinstance(
        layer,
        tf.keras.layers.SeparableConv1D
    ):
        length = int(
            layer.output.shape[1]
        )

        cin = int(
            layer.input.shape[-1]
        )

        cout = int(
            layer.filters
        )

        kernel = int(
            layer.kernel_size[0]
        )

        dm = int(
            layer.depth_multiplier
        )

        depthwise = (
            length
            * cin
            * kernel
            * dm
        )

        pointwise = (
            length
            * cin
            * dm
            * cout
        )

        return (
            depthwise
            + pointwise
        )

    if isinstance(
        layer,
        tf.keras.layers.Dense
    ):
        return (
            int(
                layer.input.shape[-1]
            )
            * int(
                layer.units
            )
        )

    return 0


def quantizable_parameter_count(layer):
    weights = layer.get_weights()

    if len(weights) == 0:
        return 0

    if isinstance(
        layer,
        tf.keras.layers.SeparableConv1D
    ):
        return (
            int(
                np.prod(
                    weights[0].shape
                )
            )
            +
            int(
                np.prod(
                    weights[1].shape
                )
            )
        )

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv1D,
            tf.keras.layers.Dense
        )
    ):
        return int(
            np.prod(
                weights[0].shape
            )
        )

    return 0


def fixed_bias_bits(layer):
    weights = layer.get_weights()

    if len(weights) == 0:
        return 0

    if isinstance(
        layer,
        tf.keras.layers.SeparableConv1D
    ):
        if len(weights) > 2:
            return (
                int(
                    np.prod(
                        weights[2].shape
                    )
                )
                * 32
            )

        return 0

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv1D,
            tf.keras.layers.Dense
        )
    ):
        if len(weights) > 1:
            return (
                int(
                    np.prod(
                        weights[1].shape
                    )
                )
                * 32
            )

    return 0


def estimate_layerwise_cost(
    model,
    precision_config,
    default_bits=8
):
    total_weight_bits = 0
    total_bitops = 0

    for layer in model.layers:
        if not isinstance(
            layer,
            WEIGHT_LAYER_TYPES
        ):
            continue

        bits = get_weight_bits(
            layer.name,
            precision_config,
            default_bits
        )

        params = (
            quantizable_parameter_count(
                layer
            )
        )

        total_weight_bits += (
            params * bits
        )

        total_weight_bits += (
            fixed_bias_bits(
                layer
            )
        )

        macs = layer_macs(
            layer
        )

        # Proxy: Wbits × Abits,
        # hiện tại layer dùng cùng precision W/A.
        total_bitops += (
            macs
            * bits
            * bits
        )

    memory_kb = (
        total_weight_bits
        / 8.0
        / 1024.0
    )

    return {
        "weight_memory_kb":
            float(memory_kb),

        "total_weight_bits":
            int(total_weight_bits),

        "bitops":
            int(total_bitops)
    }


def config_signature(
    precision_config,
    layer_names
):
    return tuple(
        int(
            precision_config.get(
                name,
                8
            )
        )
        for name
        in layer_names
    )