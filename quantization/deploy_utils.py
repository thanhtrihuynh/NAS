from pathlib import Path

import numpy as np
import tensorflow as tf

from quantization.qat_layers import (
    QATWeightWrapper,
    QATActivationWrapper
)


def unwrap_layer(layer):
    if isinstance(
        layer,
        (
            QATWeightWrapper,
            QATActivationWrapper
        )
    ):
        return layer.layer

    return layer


def get_original_layer_name(layer):
    return unwrap_layer(layer).name


def symmetric_scale(
    values,
    bits
):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    if bits == 8:
        qmax = 127.0
    elif bits == 4:
        qmax = 7.0
    else:
        raise ValueError(
            "Chỉ hỗ trợ INT4 hoặc INT8"
        )

    max_abs = float(
        np.max(
            np.abs(values)
        )
    )

    if max_abs < 1e-12:
        return 1.0

    return max_abs / qmax


def quantize_symmetric(
    values,
    bits,
    scale=None
):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    if bits == 8:
        qmin = -127
        qmax = 127
        dtype = np.int8

    elif bits == 4:
        qmin = -7
        qmax = 7
        dtype = np.int8

    else:
        raise ValueError(
            "Chỉ hỗ trợ INT4 hoặc INT8"
        )

    if scale is None:
        scale = symmetric_scale(
            values,
            bits
        )

    q = np.round(
        values / scale
    )

    q = np.clip(
        q,
        qmin,
        qmax
    )

    return (
        q.astype(dtype),
        float(scale)
    )


def dequantize(
    q,
    scale
):
    return (
        q.astype(np.float32)
        * float(scale)
    )


def fold_conv_bn(
    kernel,
    bias,
    gamma,
    beta,
    moving_mean,
    moving_variance,
    epsilon
):
    if bias is None:
        bias = np.zeros(
            gamma.shape,
            dtype=np.float32
        )

    factor = (
        gamma
        /
        np.sqrt(
            moving_variance
            + epsilon
        )
    )

    folded_kernel = (
        kernel
        * factor.reshape(
            (1, 1, -1)
        )
    )

    folded_bias = (
        beta
        +
        (
            bias
            - moving_mean
        )
        * factor
    )

    return (
        folded_kernel.astype(
            np.float32
        ),
        folded_bias.astype(
            np.float32
        )
    )


def fold_separable_conv_bn(
    depthwise_kernel,
    pointwise_kernel,
    bias,
    gamma,
    beta,
    moving_mean,
    moving_variance,
    epsilon
):
    if bias is None:
        bias = np.zeros(
            gamma.shape,
            dtype=np.float32
        )

    factor = (
        gamma
        /
        np.sqrt(
            moving_variance
            + epsilon
        )
    )

    folded_pointwise = (
        pointwise_kernel
        * factor.reshape(
            (1, 1, -1)
        )
    )

    folded_bias = (
        beta
        +
        (
            bias
            - moving_mean
        )
        * factor
    )

    return (
        depthwise_kernel.astype(
            np.float32
        ),
        folded_pointwise.astype(
            np.float32
        ),
        folded_bias.astype(
            np.float32
        )
    )


def fold_dense_bn(
    kernel,
    bias,
    gamma,
    beta,
    moving_mean,
    moving_variance,
    epsilon
):
    if bias is None:
        bias = np.zeros(
            gamma.shape,
            dtype=np.float32
        )

    factor = (
        gamma
        /
        np.sqrt(
            moving_variance
            + epsilon
        )
    )

    folded_kernel = (
        kernel
        * factor.reshape(
            (1, -1)
        )
    )

    folded_bias = (
        beta
        +
        (
            bias
            - moving_mean
        )
        * factor
    )

    return (
        folded_kernel.astype(
            np.float32
        ),
        folded_bias.astype(
            np.float32
        )
    )