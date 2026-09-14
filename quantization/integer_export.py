import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from quantization.qat_layers import (
    QATWeightWrapper
)

from quantization.deploy_utils import (
    quantize_symmetric,
    fold_conv_bn,
    fold_separable_conv_bn,
    fold_dense_bn
)


def get_inner_weight_layers(
    model
):
    result = {}

    for layer in model.layers:
        if isinstance(
            layer,
            QATWeightWrapper
        ):
            result[
                layer.layer.name
            ] = layer.layer

    return result


def get_bn_layers(
    model
):
    result = {}

    for layer in model.layers:
        inner = getattr(
            layer,
            "layer",
            layer
        )

        if isinstance(
            inner,
            tf.keras.layers.BatchNormalization
        ):
            result[
                inner.name
            ] = inner

    return result


def guess_bn_name(
    layer_name
):
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
        ),
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

    if layer_name == "stem_conv":
        return "stem_bn"

    if layer_name == "head_dense":
        return "head_bn"

    return None


def get_bn_parameters(
    bn
):
    gamma = (
        bn.gamma.numpy()
        if bn.scale
        else np.ones(
            bn.moving_mean.shape,
            dtype=np.float32
        )
    )

    beta = (
        bn.beta.numpy()
        if bn.center
        else np.zeros(
            bn.moving_mean.shape,
            dtype=np.float32
        )
    )

    return (
        gamma,
        beta,
        bn.moving_mean.numpy(),
        bn.moving_variance.numpy(),
        float(bn.epsilon)
    )


def export_integer_parameters(
    model,
    precision_config,
    activation_scales,
    export_dir
):
    export_dir = Path(
        export_dir
    )

    export_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    weight_layers = (
        get_inner_weight_layers(
            model
        )
    )

    bn_layers = (
        get_bn_layers(
            model
        )
    )

    metadata = {}

    for layer_name, layer in (
        weight_layers.items()
    ):
        bits = int(
            precision_config[
                layer_name
            ]
        )

        weights = (
            layer.get_weights()
        )

        bn_name = guess_bn_name(
            layer_name
        )

        bn = (
            bn_layers.get(
                bn_name
            )
            if bn_name is not None
            else None
        )

        layer_meta = {
            "type":
                layer.__class__.__name__,

            "weight_bits":
                bits,

            "bn_folded":
                bn is not None
        }

        if isinstance(
            layer,
            tf.keras.layers.Conv1D
        ):
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
                    var,
                    eps
                ) = get_bn_parameters(
                    bn
                )

                kernel, bias = (
                    fold_conv_bn(
                        kernel,
                        bias,
                        gamma,
                        beta,
                        mean,
                        var,
                        eps
                    )
                )

            if bias is None:
                bias = np.zeros(
                    kernel.shape[-1],
                    dtype=np.float32
                )

            q_kernel, w_scale = (
                quantize_symmetric(
                    kernel,
                    bits
                )
            )

            np.save(
                export_dir
                / f"{layer_name}_kernel.npy",
                q_kernel
            )

            np.save(
                export_dir
                / f"{layer_name}_bias_float.npy",
                bias.astype(
                    np.float32
                )
            )

            layer_meta[
                "weight_scale"
            ] = w_scale

            layer_meta[
                "kernel_shape"
            ] = list(
                q_kernel.shape
            )

        elif isinstance(
            layer,
            tf.keras.layers.SeparableConv1D
        ):
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
                    var,
                    eps
                ) = get_bn_parameters(
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
                        var,
                        eps
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

            np.save(
                export_dir
                / (
                    f"{layer_name}"
                    "_depthwise.npy"
                ),
                q_depthwise
            )

            np.save(
                export_dir
                / (
                    f"{layer_name}"
                    "_pointwise.npy"
                ),
                q_pointwise
            )

            np.save(
                export_dir
                / (
                    f"{layer_name}"
                    "_bias_float.npy"
                ),
                bias.astype(
                    np.float32
                )
            )

            layer_meta[
                "depthwise_scale"
            ] = (
                depthwise_scale
            )

            layer_meta[
                "pointwise_scale"
            ] = (
                pointwise_scale
            )

            layer_meta[
                "depthwise_shape"
            ] = list(
                q_depthwise.shape
            )

            layer_meta[
                "pointwise_shape"
            ] = list(
                q_pointwise.shape
            )

        elif isinstance(
            layer,
            tf.keras.layers.Dense
        ):
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
                    var,
                    eps
                ) = get_bn_parameters(
                    bn
                )

                kernel, bias = (
                    fold_dense_bn(
                        kernel,
                        bias,
                        gamma,
                        beta,
                        mean,
                        var,
                        eps
                    )
                )

            if bias is None:
                bias = np.zeros(
                    kernel.shape[-1],
                    dtype=np.float32
                )

            q_kernel, w_scale = (
                quantize_symmetric(
                    kernel,
                    bits
                )
            )

            np.save(
                export_dir
                / f"{layer_name}_kernel.npy",
                q_kernel
            )

            np.save(
                export_dir
                / f"{layer_name}_bias_float.npy",
                bias.astype(
                    np.float32
                )
            )

            layer_meta[
                "weight_scale"
            ] = w_scale

            layer_meta[
                "kernel_shape"
            ] = list(
                q_kernel.shape
            )

        metadata[
            layer_name
        ] = layer_meta

    metadata[
        "_activation_scales"
    ] = activation_scales

    return metadata