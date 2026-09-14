import numpy as np
import tensorflow as tf

from quantization.qat_layers import (
    QATWeightWrapper,
    QATActivationWrapper
)

from quantization.fixed_point import (
    quantize_symmetric
)


def canonical_name(layer):
    if isinstance(
        layer,
        (
            QATWeightWrapper,
            QATActivationWrapper
        )
    ):
        return layer.layer.name

    return layer.name


def output_bits_for_layer(
    layer_name,
    precision_config
):
    if layer_name == "stem_act":
        return int(
            precision_config[
                "stem_conv"
            ]
        )

    if layer_name == "head_act":
        return int(
            precision_config[
                "head_dense"
            ]
        )

    mappings = [
        (
            "_b1_act",
            "_b1_sepconv"
        ),
        (
            "_b2_act",
            "_b2_sepconv"
        ),
        (
            "_b3_act",
            "_b3_sepconv"
        ),
        (
            "_pool_act",
            "_pool_proj"
        ),
        (
            "_out_bn",
            "_out_proj"
        ),
        (
            "_shortcut_bn",
            "_shortcut"
        )
    ]

    for suffix, weight_suffix in mappings:
        if layer_name.endswith(
            suffix
        ):
            prefix = layer_name[
                :-len(suffix)
            ]

            weight_name = (
                prefix
                + weight_suffix
            )

            if weight_name in (
                precision_config
            ):
                return int(
                    precision_config[
                        weight_name
                    ]
                )

    # Các điểm ghép residual, concat,
    # pooling và boundary giữ activation INT8.
    return 8


def collect_graph_ranges(
    model,
    X_calibration,
    precision_config,
    batch_size=128
):
    probe_layers = []

    for layer in model.layers:
        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):
            continue

        try:
            _ = layer.output

        except Exception:
            continue

        probe_layers.append(
            layer
        )

    if len(probe_layers) == 0:
        raise RuntimeError(
            "Không tìm thấy layer để calibration."
        )

    probe_model = (
        tf.keras.Model(
            inputs=model.input,
            outputs=[
                layer.output
                for layer
                in probe_layers
            ]
        )
    )

    max_abs = {}

    for layer in probe_layers:
        name = canonical_name(
            layer
        )

        max_abs[name] = 0.0

    input_max_abs = float(
        np.max(
            np.abs(
                X_calibration
            )
        )
    )

    total = len(
        X_calibration
    )

    for start in range(
        0,
        total,
        batch_size
    ):
        end = min(
            start + batch_size,
            total
        )

        print(
            f"Graph calibration: "
            f"{start}/{total}",
            end="\r"
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
            probe_layers,
            values
        ):
            name = canonical_name(
                layer
            )

            value = float(
                tf.reduce_max(
                    tf.abs(
                        tensor
                    )
                ).numpy()
            )

            if value > max_abs[name]:
                max_abs[name] = value

    print()

    scales = {}

    # Input ECG luôn A8.
    input_scale = (
        input_max_abs / 127.0
        if input_max_abs > 1e-12
        else 1.0
    )

    scales[
        "__input__"
    ] = {
        "bits": 8,
        "max_abs":
            input_max_abs,
        "scale":
            float(
                input_scale
            )
    }

    for name, maximum in (
        max_abs.items()
    ):
        bits = output_bits_for_layer(
            name,
            precision_config
        )

        qmax = (
            127.0
            if bits == 8
            else 7.0
        )

        scale = (
            maximum / qmax
            if maximum > 1e-12
            else 1.0
        )

        scales[name] = {
            "bits":
                int(bits),

            "max_abs":
                float(
                    maximum
                ),

            "scale":
                float(
                    scale
                )
        }

    return scales


def collect_sepconv_internal_ranges(
    model,
    X_calibration,
    precision_config,
    batch_size=128
):
    """
    Thu thập range đầu ra Depthwise trước Pointwise
    cho từng SeparableConv1D.

    FPGA dự kiến:

        input integer
            ↓
        Depthwise Conv
            ↓
        INT32 accumulator
            ↓
        requantization
            ↓
        INT4/INT8 intermediate
            ↓
        Pointwise Conv
    """

    sep_wrappers = []

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
            sep_wrappers.append(
                layer
            )

    if len(
        sep_wrappers
    ) == 0:
        print(
            "Không tìm thấy SeparableConv1D."
        )

        return {}

    print(
        "SeparableConv layers:",
        len(sep_wrappers)
    )

    # Lấy tensor input thực tế
    # đi vào từng SeparableConv1D.
    probe_model = (
        tf.keras.Model(
            inputs=model.input,
            outputs=[
                wrapper.input
                for wrapper
                in sep_wrappers
            ]
        )
    )

    internal_max = {
        wrapper.layer.name: 0.0
        for wrapper
        in sep_wrappers
    }

    depthwise_kernels = {}

    # Weight được fake-quant trước khi
    # tính range intermediate.
    for wrapper in sep_wrappers:
        layer = wrapper.layer

        bits = int(
            precision_config[
                layer.name
            ]
        )

        kernel = (
            layer.depthwise_kernel
            .numpy()
        )

        (
            q_kernel,
            kernel_scale
        ) = quantize_symmetric(
            kernel,
            bits
        )

        dequantized_kernel = (
            q_kernel.astype(
                np.float32
            )
            * float(
                kernel_scale
            )
        )

        depthwise_kernels[
            layer.name
        ] = tf.convert_to_tensor(
            dequantized_kernel,
            dtype=tf.float32
        )

    total = len(
        X_calibration
    )

    for start in range(
        0,
        total,
        batch_size
    ):
        end = min(
            start + batch_size,
            total
        )

        print(
            f"SepConv calibration: "
            f"{start}/{total}",
            end="\r"
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
            sep_wrappers,
            inputs
        ):
            layer = wrapper.layer

            kernel = (
                depthwise_kernels[
                    layer.name
                ]
            )

            x = tf.cast(
                x,
                tf.float32
            )

            stride = int(
                layer.strides[0]
            )

            dilation = int(
                layer.dilation_rate[0]
            )

            # Keras 3 hỗ trợ
            # Depthwise Conv 1D trực tiếp.
            y = (
                tf.keras.ops.depthwise_conv(
                    x,
                    kernel,
                    strides=stride,
                    padding=
                        layer.padding,
                    data_format=
                        (
                            layer.data_format
                            or "channels_last"
                        ),
                    dilation_rate=
                        dilation
                )
            )

            maximum = float(
                tf.reduce_max(
                    tf.abs(y)
                ).numpy()
            )

            if (
                maximum
                >
                internal_max[
                    layer.name
                ]
            ):
                internal_max[
                    layer.name
                ] = maximum

    print()

    result = {}

    for wrapper in sep_wrappers:
        name = (
            wrapper.layer.name
        )

        bits = int(
            precision_config[
                name
            ]
        )

        qmax = (
            127.0
            if bits == 8
            else 7.0
        )

        maximum = (
            internal_max[
                name
            ]
        )

        scale = (
            maximum / qmax
            if maximum > 1e-12
            else 1.0
        )

        result[name] = {
            "bits":
                bits,

            "max_abs":
                float(
                    maximum
                ),

            "scale":
                float(
                    scale
                )
        }

        print(
            f"{name:<30} "
            f"INT{bits:<2} "
            f"max_abs="
            f"{maximum:.8f} "
            f"scale="
            f"{scale:.10f}"
        )

    return result