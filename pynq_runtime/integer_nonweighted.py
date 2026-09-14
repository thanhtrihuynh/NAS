import math

import numpy as np


def qrange(bits):
    if bits == 4:
        return -7, 7

    if bits == 8:
        return -127, 127

    raise ValueError(
        f"Unsupported bits={bits}"
    )


def quantize(
    x,
    scale,
    bits,
):
    qmin, qmax = qrange(
        bits
    )

    q = np.rint(
        np.asarray(
            x,
            dtype=np.float64,
        )
        / float(scale)
    )

    return np.clip(
        q,
        qmin,
        qmax,
    ).astype(np.int8)


def dequantize(
    q,
    scale,
):
    return (
        np.asarray(
            q,
            dtype=np.float64,
        )
        * float(scale)
    )


def gelu_requant(
    q_in,
    input_scale,
    output_scale,
    output_bits,
):
    x = dequantize(
        q_in,
        input_scale,
    )

    erf = np.vectorize(
        math.erf,
        otypes=[np.float64],
    )

    y = (
        0.5
        * x
        * (
            1.0
            + erf(
                x
                / math.sqrt(2.0)
            )
        )
    )

    return quantize(
        y,
        output_scale,
        output_bits,
    )


def requantize_tensor(
    q,
    input_scale,
    output_scale,
    bits,
):
    return quantize(
        dequantize(
            q,
            input_scale,
        ),
        output_scale,
        bits,
    )


def maxpool1d_same3_s1(x):
    x = np.asarray(x)
    length, channels = x.shape

    out = np.empty_like(
        x
    )

    for t in range(length):
        start = max(0, t - 1)
        end = min(
            length,
            t + 2,
        )

        out[t] = np.max(
            x[start:end],
            axis=0,
        )

    return out


def maxpool1d_2_s2(x):
    x = np.asarray(x)

    length, channels = x.shape

    if length % 2 != 0:
        raise ValueError(
            "Pool2 stride2 yêu cầu length chẵn."
        )

    out = np.empty(
        (
            length // 2,
            channels,
        ),
        dtype=x.dtype,
    )

    for i in range(
        length // 2
    ):
        out[i] = np.maximum(
            x[2 * i],
            x[2 * i + 1],
        )

    return out


def residual_add_scaled(
    main_q,
    main_scale,
    shortcut_q,
    shortcut_scale,
    output_scale,
    output_bits,
):
    main = requantize_tensor(
        main_q,
        main_scale,
        output_scale,
        output_bits,
    ).astype(np.int16)

    shortcut = requantize_tensor(
        shortcut_q,
        shortcut_scale,
        output_scale,
        output_bits,
    ).astype(np.int16)

    qmin, qmax = qrange(
        output_bits
    )

    return np.clip(
        main + shortcut,
        qmin,
        qmax,
    ).astype(np.int8)


def global_average_requant(
    x,
    input_scale,
    output_scale,
    output_bits,
):
    real = dequantize(
        x,
        input_scale,
    )

    avg = np.mean(
        real,
        axis=0,
    )

    return quantize(
        avg,
        output_scale,
        output_bits,
    )


def global_max_requant(
    x,
    input_scale,
    output_scale,
    output_bits,
):
    maximum = np.max(
        np.asarray(x),
        axis=0,
    )

    return requantize_tensor(
        maximum,
        input_scale,
        output_scale,
        output_bits,
    )


def dense_logits_int32(
    x,
    kernel,
    bias,
):
    x = np.asarray(
        x,
        dtype=np.int8,
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int8,
    )

    bias = np.asarray(
        bias,
        dtype=np.int32,
    )

    return (
        x.astype(np.int32)
        @ kernel.astype(np.int32)
        + bias
    ).astype(np.int32)


def softmax_float(logits):
    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    shifted = logits - np.max(
        logits
    )

    exp = np.exp(
        shifted
    )

    return exp / np.sum(exp)
