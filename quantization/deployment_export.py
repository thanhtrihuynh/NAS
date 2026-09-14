import re
from pathlib import Path

import numpy as np
import tensorflow as tf

from quantization.qat_layers import (
    QATWeightWrapper
)

from quantization.deploy_utils import (
    fold_conv_bn,
    fold_separable_conv_bn,
    fold_dense_bn
)

from quantization.fixed_point import (
    quantize_symmetric,
    quantize_bias_int32,
    multiplier_shift
)


def get_inner_layers(model):
    weight_layers = {}
    bn_layers = {}

    for layer in model.layers:
        inner = getattr(
            layer,
            "layer",
            layer
        )

        if isinstance(
            layer,
            QATWeightWrapper
        ):
            weight_layers[
                inner.name
            ] = inner

        if isinstance(
            inner,
            tf.keras.layers.BatchNormalization
        ):
            bn_layers[
                inner.name
            ] = inner

    return (
        weight_layers,
        bn_layers
    )


def bn_name_for_layer(
    layer_name
):
    if layer_name == "stem_conv":
        return "stem_bn"

    if layer_name == "head_dense":
        return "head_bn"

    replacements = [
        (
            "_b1_sepconv",
            "_b1_bn"
        ),
        (
            "_b2_sepconv",
            "_b2_bn"
        ),
        (
            "_b3_sepconv",
            "_b3_bn"
        ),
        (
            "_pool_proj",
            "_pool_bn"
        ),
        (
            "_out_proj",
            "_out_bn"
        ),
        (
            "_shortcut",
            "_shortcut_bn"
        )
    ]

    for source, target in replacements:
        if layer_name.endswith(
            source
        ):
            return (
                layer_name[
                    :-len(source)
                ]
                + target
            )

    return None


def activation_name_for_layer(
    layer_name
):
    if layer_name == "stem_conv":
        return "stem_act"

    if layer_name == "head_dense":
        return "head_act"

    replacements = [
        (
            "_b1_sepconv",
            "_b1_act"
        ),
        (
            "_b2_sepconv",
            "_b2_act"
        ),
        (
            "_b3_sepconv",
            "_b3_act"
        ),
        (
            "_pool_proj",
            "_pool_act"
        )
    ]

    for source, target in replacements:
        if layer_name.endswith(
            source
        ):
            return (
                layer_name[
                    :-len(source)
                ]
                + target
            )

    return None


def input_source_for_layer(
    layer_name
):
    if layer_name == "stem_conv":
        return "__input__"

    if layer_name == "head_dense":
        return "head_concat"

    if layer_name == "class_output":
        return "head_act"

    match = re.match(
        r"block([1-4])_(.+)",
        layer_name
    )

    if match is None:
        raise ValueError(
            "Không xác định được input source "
            f"của {layer_name}"
        )

    block = int(
        match.group(1)
    )

    suffix = match.group(2)

    # out_proj nhận kết quả concat
    # của 4 nhánh trong chính block.
    if suffix == "out_proj":
        return (
            f"block{block}_concat"
        )

    # Các branch, pool projection và shortcut
    # nhận input chung của block.
    if block == 1:
        return "stem_act"

    if block == 2:
        return "pool1"

    if block == 3:
        return "pool2"

    if block == 4:
        return "pool3"

    raise ValueError(
        f"Block không hợp lệ: {block}"
    )


def get_bn_values(
    bn
):
    if bn.scale:
        gamma = (
            bn.gamma.numpy()
        )
    else:
        gamma = np.ones(
            bn.moving_mean.shape,
            dtype=np.float32
        )

    if bn.center:
        beta = (
            bn.beta.numpy()
        )
    else:
        beta = np.zeros(
            bn.moving_mean.shape,
            dtype=np.float32
        )

    return (
        gamma.astype(
            np.float32
        ),
        beta.astype(
            np.float32
        ),
        bn.moving_mean
        .numpy()
        .astype(
            np.float32
        ),
        bn.moving_variance
        .numpy()
        .astype(
            np.float32
        ),
        float(
            bn.epsilon
        )
    )


def scale_of(
    tensor_scales,
    name
):
    if name not in tensor_scales:
        raise KeyError(
            "Không tìm thấy activation scale "
            f"cho '{name}'."
        )

    return float(
        tensor_scales[
            name
        ]["scale"]
    )


def bits_of(
    tensor_scales,
    name,
    default=8
):
    if name not in tensor_scales:
        return int(
            default
        )

    return int(
        tensor_scales[
            name
        ]["bits"]
    )


def export_conv1d(
    layer_name,
    layer,
    bits,
    input_scale,
    bn,
    export_dir
):
    weights = (
        layer.get_weights()
    )

    kernel = weights[0]

    bias = (
        weights[1]
        if layer.use_bias
        else None
    )

    if bn is not None:
        (
            gamma,
            beta,
            mean,
            variance,
            epsilon
        ) = get_bn_values(
            bn
        )

        (
            kernel,
            bias
        ) = fold_conv_bn(
            kernel,
            bias,
            gamma,
            beta,
            mean,
            variance,
            epsilon
        )

    if bias is None:
        bias = np.zeros(
            kernel.shape[-1],
            dtype=np.float32
        )

    (
        q_kernel,
        weight_scale
    ) = quantize_symmetric(
        kernel,
        bits
    )

    accumulator_scale = (
        input_scale
        * weight_scale
    )

    q_bias = (
        quantize_bias_int32(
            bias,
            accumulator_scale
        )
    )

    kernel_file = (
        f"{layer_name}_kernel.npy"
    )

    bias_file = (
        f"{layer_name}_bias_int32.npy"
    )

    np.save(
        export_dir
        / kernel_file,
        q_kernel
    )

    np.save(
        export_dir
        / bias_file,
        q_bias
    )

    return {
        "type":
            "Conv1D",

        "bits":
            int(bits),

        "input_scale":
            float(
                input_scale
            ),

        "weight_scale":
            float(
                weight_scale
            ),

        "accumulator_scale":
            float(
                accumulator_scale
            ),

        "kernel_file":
            kernel_file,

        "bias_file":
            bias_file,

        "kernel_shape":
            list(
                q_kernel.shape
            ),

        "stride":
            int(
                layer.strides[0]
            ),

        "dilation":
            int(
                layer.dilation_rate[0]
            ),

        "padding":
            layer.padding,

        "bn_folded":
            bn is not None
    }


def export_separable_conv1d(
    layer_name,
    layer,
    bits,
    input_scale,
    bn,
    sep_internal,
    export_dir
):
    weights = (
        layer.get_weights()
    )

    depthwise = (
        weights[0]
    )

    pointwise = (
        weights[1]
    )

    bias = (
        weights[2]
        if layer.use_bias
        else None
    )

    if bn is not None:
        (
            gamma,
            beta,
            mean,
            variance,
            epsilon
        ) = get_bn_values(
            bn
        )

        (
            depthwise,
            pointwise,
            bias
        ) = (
            fold_separable_conv_bn(
                depthwise,
                pointwise,
                bias,
                gamma,
                beta,
                mean,
                variance,
                epsilon
            )
        )

    if bias is None:
        bias = np.zeros(
            pointwise.shape[-1],
            dtype=np.float32
        )

    (
        q_depthwise,
        depthwise_scale
    ) = quantize_symmetric(
        depthwise,
        bits
    )

    (
        q_pointwise,
        pointwise_scale
    ) = quantize_symmetric(
        pointwise,
        bits
    )

    depthwise_acc_scale = (
        input_scale
        * depthwise_scale
    )

    if layer_name not in (
        sep_internal
    ):
        raise KeyError(
            "Thiếu internal scale cho "
            f"{layer_name}"
        )

    depthwise_output_scale = float(
        sep_internal[
            layer_name
        ]["scale"]
    )

    depthwise_requant_ratio = (
        depthwise_acc_scale
        / depthwise_output_scale
    )

    depthwise_requant = (
        multiplier_shift(
            depthwise_requant_ratio
        )
    )

    pointwise_acc_scale = (
        depthwise_output_scale
        * pointwise_scale
    )

    q_bias = (
        quantize_bias_int32(
            bias,
            pointwise_acc_scale
        )
    )

    depthwise_file = (
        f"{layer_name}_depthwise.npy"
    )

    pointwise_file = (
        f"{layer_name}_pointwise.npy"
    )

    bias_file = (
        f"{layer_name}_bias_int32.npy"
    )

    np.save(
        export_dir
        / depthwise_file,
        q_depthwise
    )

    np.save(
        export_dir
        / pointwise_file,
        q_pointwise
    )

    np.save(
        export_dir
        / bias_file,
        q_bias
    )

    return {
        "type":
            "SeparableConv1D",

        "bits":
            int(bits),

        "input_scale":
            float(
                input_scale
            ),

        "depthwise_scale":
            float(
                depthwise_scale
            ),

        "depthwise_acc_scale":
            float(
                depthwise_acc_scale
            ),

        "depthwise_output_scale":
            float(
                depthwise_output_scale
            ),

        "depthwise_requant":
            depthwise_requant,

        "pointwise_scale":
            float(
                pointwise_scale
            ),

        "pointwise_acc_scale":
            float(
                pointwise_acc_scale
            ),

        "depthwise_file":
            depthwise_file,

        "pointwise_file":
            pointwise_file,

        "bias_file":
            bias_file,

        "depthwise_shape":
            list(
                q_depthwise.shape
            ),

        "pointwise_shape":
            list(
                q_pointwise.shape
            ),

        "stride":
            int(
                layer.strides[0]
            ),

        "dilation":
            int(
                layer.dilation_rate[0]
            ),

        "padding":
            layer.padding,

        "bn_folded":
            bn is not None
    }


def export_dense(
    layer_name,
    layer,
    bits,
    input_scale,
    bn,
    export_dir
):
    weights = (
        layer.get_weights()
    )

    kernel = (
        weights[0]
    )

    bias = (
        weights[1]
        if layer.use_bias
        else None
    )

    if bn is not None:
        (
            gamma,
            beta,
            mean,
            variance,
            epsilon
        ) = get_bn_values(
            bn
        )

        (
            kernel,
            bias
        ) = fold_dense_bn(
            kernel,
            bias,
            gamma,
            beta,
            mean,
            variance,
            epsilon
        )

    if bias is None:
        bias = np.zeros(
            kernel.shape[-1],
            dtype=np.float32
        )

    (
        q_kernel,
        weight_scale
    ) = quantize_symmetric(
        kernel,
        bits
    )

    accumulator_scale = (
        input_scale
        * weight_scale
    )

    q_bias = (
        quantize_bias_int32(
            bias,
            accumulator_scale
        )
    )

    kernel_file = (
        f"{layer_name}_kernel.npy"
    )

    bias_file = (
        f"{layer_name}_bias_int32.npy"
    )

    np.save(
        export_dir
        / kernel_file,
        q_kernel
    )

    np.save(
        export_dir
        / bias_file,
        q_bias
    )

    return {
        "type":
            "Dense",

        "bits":
            int(bits),

        "input_scale":
            float(
                input_scale
            ),

        "weight_scale":
            float(
                weight_scale
            ),

        "accumulator_scale":
            float(
                accumulator_scale
            ),

        "kernel_file":
            kernel_file,

        "bias_file":
            bias_file,

        "kernel_shape":
            list(
                q_kernel.shape
            ),

        "bn_folded":
            bn is not None
    }


def export_deployment_model(
    model,
    precision_config,
    scale_data,
    export_dir
):
    export_dir = Path(
        export_dir
    )

    export_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        weight_layers,
        bn_layers
    ) = get_inner_layers(
        model
    )

    tensor_scales = (
        scale_data[
            "tensor_scales"
        ]
    )

    sep_internal = (
        scale_data[
            "separable_internal_scales"
        ]
    )

    manifest = {
        "format_version":
            2,

        "quantization":
            "symmetric_per_tensor",

        "weight_precision":
            "layer_wise_int4_int8",

        "accumulator_bits":
            32,

        "softmax":
            "PS_ARM_float",

        "layers":
            {}
    }

    for (
        layer_name,
        layer
    ) in weight_layers.items():

        print(
            "Export:",
            layer_name
        )

        if layer_name not in (
            precision_config
        ):
            raise KeyError(
                "Không tìm thấy precision cho "
                f"{layer_name}"
            )

        bits = int(
            precision_config[
                layer_name
            ]
        )

        input_source = (
            input_source_for_layer(
                layer_name
            )
        )

        input_scale = scale_of(
            tensor_scales,
            input_source
        )

        bn_name = (
            bn_name_for_layer(
                layer_name
            )
        )

        bn = (
            bn_layers.get(
                bn_name
            )
            if bn_name
            else None
        )

        if isinstance(
            layer,
            tf.keras.layers.SeparableConv1D
        ):
            info = (
                export_separable_conv1d(
                    layer_name,
                    layer,
                    bits,
                    input_scale,
                    bn,
                    sep_internal,
                    export_dir
                )
            )

        elif isinstance(
            layer,
            tf.keras.layers.Conv1D
        ):
            info = (
                export_conv1d(
                    layer_name,
                    layer,
                    bits,
                    input_scale,
                    bn,
                    export_dir
                )
            )

        elif isinstance(
            layer,
            tf.keras.layers.Dense
        ):
            info = (
                export_dense(
                    layer_name,
                    layer,
                    bits,
                    input_scale,
                    bn,
                    export_dir
                )
            )

        else:
            continue

        info[
            "input_source"
        ] = input_source

        info[
            "bn_name"
        ] = bn_name

        activation_name = (
            activation_name_for_layer(
                layer_name
            )
        )

        info[
            "activation_name"
        ] = activation_name

        if (
            layer_name
            == "class_output"
        ):
            info[
                "logit_scale"
            ] = float(
                info[
                    "accumulator_scale"
                ]
            )

            info[
                "output_mode"
            ] = "INT32_LOGITS"

        elif activation_name is not None:
            preactivation_name = (
                bn_name
                if bn_name is not None
                else layer_name
            )

            preactivation_scale = (
                scale_of(
                    tensor_scales,
                    preactivation_name
                )
            )

            output_scale = (
                scale_of(
                    tensor_scales,
                    activation_name
                )
            )

            if (
                info["type"]
                == "SeparableConv1D"
            ):
                source_acc_scale = (
                    info[
                        "pointwise_acc_scale"
                    ]
                )
            else:
                source_acc_scale = (
                    info[
                        "accumulator_scale"
                    ]
                )

            ratio = (
                source_acc_scale
                / preactivation_scale
            )

            info[
                "preactivation_scale"
            ] = float(
                preactivation_scale
            )

            info[
                "output_scale"
            ] = float(
                output_scale
            )

            info[
                "output_bits"
            ] = bits_of(
                tensor_scales,
                activation_name,
                default=8
            )

            info[
                "acc_to_preact"
            ] = multiplier_shift(
                ratio
            )

            info[
                "output_mode"
            ] = (
                "REQUANT_GELU_REQUANT"
            )

        else:
            output_tensor_name = (
                bn_name
                if bn_name is not None
                else layer_name
            )

            output_scale = (
                scale_of(
                    tensor_scales,
                    output_tensor_name
                )
            )

            if (
                info["type"]
                == "SeparableConv1D"
            ):
                source_acc_scale = (
                    info[
                        "pointwise_acc_scale"
                    ]
                )
            else:
                source_acc_scale = (
                    info[
                        "accumulator_scale"
                    ]
                )

            ratio = (
                source_acc_scale
                / output_scale
            )

            info[
                "output_scale"
            ] = float(
                output_scale
            )

            info[
                "output_bits"
            ] = bits_of(
                tensor_scales,
                output_tensor_name,
                default=8
            )

            info[
                "acc_to_output"
            ] = multiplier_shift(
                ratio
            )

            info[
                "output_mode"
            ] = "REQUANT"

        manifest[
            "layers"
        ][
            layer_name
        ] = info

    manifest[
        "tensor_scales"
    ] = tensor_scales

    manifest[
        "separable_internal_scales"
    ] = sep_internal

    return manifest