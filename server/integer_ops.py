import math

import numpy as np


# ============================================================
# QUANTIZATION BASIC
# ============================================================

def qrange(bits):
    if bits == 4:
        return -7, 7

    if bits == 8:
        return -127, 127

    raise ValueError(
        f"Unsupported bits={bits}"
    )


def saturate(
    value,
    bits
):
    qmin, qmax = qrange(
        bits
    )

    return max(
        qmin,
        min(
            qmax,
            int(value)
        )
    )


def clip_tensor(
    x,
    bits
):
    qmin, qmax = qrange(
        bits
    )

    return np.clip(
        x,
        qmin,
        qmax
    ).astype(
        np.int8
    )


def round_shift_signed(
    value,
    shift
):
    value = int(value)
    shift = int(shift)

    if shift == 0:
        return value

    bias = (
        1
        << (shift - 1)
    )

    if value >= 0:
        return (
            value
            + bias
        ) >> shift

    return -(
        (
            (-value)
            + bias
        )
        >> shift
    )


def quantize(
    x,
    scale,
    bits
):
    qmin, qmax = qrange(
        bits
    )

    q = np.rint(
        np.asarray(
            x,
            dtype=np.float64
        )
        / float(scale)
    )

    q = np.clip(
        q,
        qmin,
        qmax
    )

    return q.astype(
        np.int8
    )


def dequantize(
    q,
    scale
):
    return (
        np.asarray(
            q,
            dtype=np.float64
        )
        * float(scale)
    )


# ============================================================
# REQUANTIZATION BETWEEN TENSOR SCALES
# ============================================================

def requantize_scale(
    q,
    src_scale,
    dst_scale,
    bits
):
    x = dequantize(
        q,
        src_scale
    )

    return quantize(
        x,
        dst_scale,
        bits
    )


def requantize_tensor(
    q,
    input_scale,
    output_scale,
    bits
):
    return requantize_scale(
        q=q,
        src_scale=input_scale,
        dst_scale=output_scale,
        bits=bits
    )


# ============================================================
# GELU
# ============================================================

def gelu_float(
    x
):
    x = np.asarray(
        x,
        dtype=np.float64
    )

    erf_func = np.vectorize(
        math.erf,
        otypes=[
            np.float64
        ]
    )

    return (
        0.5
        * x
        * (
            1.0
            + erf_func(
                x
                / math.sqrt(2.0)
            )
        )
    )


def gelu_requant(
    q_in,
    input_scale,
    output_scale,
    output_bits
):
    x = dequantize(
        q_in,
        input_scale
    )

    y = gelu_float(
        x
    )

    return quantize(
        y,
        output_scale,
        output_bits
    )


# ============================================================
# CONV1D
# ============================================================

def conv1d_same(
    x,
    kernel,
    bias,
    multiplier,
    shift,
    output_bits,
    dilation=1
):
    x = np.asarray(
        x,
        dtype=np.int8
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int8
    )

    bias = np.asarray(
        bias,
        dtype=np.int32
    )

    if x.ndim != 2:
        raise ValueError(
            "Conv1D input phải có "
            f"shape [L,C], nhận {x.shape}"
        )

    if kernel.ndim != 3:
        raise ValueError(
            "Conv1D kernel phải có "
            f"shape [K,Cin,Cout], nhận {kernel.shape}"
        )

    length, cin = x.shape

    kernel_size, kernel_cin, cout = (
        kernel.shape
    )

    if cin != kernel_cin:
        raise ValueError(
            "Conv1D Cin mismatch: "
            f"input={cin}, kernel={kernel_cin}"
        )

    if bias.shape != (
        cout,
    ):
        raise ValueError(
            "Conv1D bias mismatch: "
            f"{bias.shape}, Cout={cout}"
        )

    pad_left = (
        dilation
        * (
            kernel_size - 1
        )
    ) // 2

    output = np.zeros(
        (
            length,
            cout
        ),
        dtype=np.int8
    )

    for out_pos in range(
        length
    ):
        for out_ch in range(
            cout
        ):
            acc = int(
                bias[out_ch]
            )

            for k in range(
                kernel_size
            ):
                input_pos = (
                    out_pos
                    + k * dilation
                    - pad_left
                )

                if (
                    input_pos < 0
                    or
                    input_pos >= length
                ):
                    continue

                for in_ch in range(
                    cin
                ):
                    acc += (
                        int(
                            x[
                                input_pos,
                                in_ch
                            ]
                        )
                        *
                        int(
                            kernel[
                                k,
                                in_ch,
                                out_ch
                            ]
                        )
                    )

            product = (
                acc
                * int(multiplier)
            )

            value = round_shift_signed(
                product,
                shift
            )

            output[
                out_pos,
                out_ch
            ] = saturate(
                value,
                output_bits
            )

    return output


# ============================================================
# DEPTHWISE CONV1D
# ============================================================

def depthwise_conv1d_same(
    x,
    kernel,
    multiplier,
    shift,
    output_bits,
    dilation=1
):
    x = np.asarray(
        x,
        dtype=np.int8
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int8
    )

    if x.ndim != 2:
        raise ValueError(
            "Depthwise input phải có "
            f"shape [L,C], nhận {x.shape}"
        )

    if kernel.ndim != 3:
        raise ValueError(
            "Depthwise kernel phải có "
            f"shape [K,Cin,1], nhận {kernel.shape}"
        )

    length, cin = x.shape

    kernel_size, kernel_cin, dm = (
        kernel.shape
    )

    if cin != kernel_cin:
        raise ValueError(
            "Depthwise Cin mismatch: "
            f"input={cin}, kernel={kernel_cin}"
        )

    if dm != 1:
        raise ValueError(
            "Hiện chỉ hỗ trợ "
            "depth_multiplier = 1"
        )

    pad_left = (
        dilation
        * (
            kernel_size - 1
        )
    ) // 2

    output = np.zeros(
        (
            length,
            cin
        ),
        dtype=np.int8
    )

    for out_pos in range(
        length
    ):
        for channel in range(
            cin
        ):
            acc = 0

            for k in range(
                kernel_size
            ):
                input_pos = (
                    out_pos
                    + k * dilation
                    - pad_left
                )

                if (
                    input_pos < 0
                    or
                    input_pos >= length
                ):
                    continue

                acc += (
                    int(
                        x[
                            input_pos,
                            channel
                        ]
                    )
                    *
                    int(
                        kernel[
                            k,
                            channel,
                            0
                        ]
                    )
                )

            product = (
                acc
                * int(multiplier)
            )

            value = round_shift_signed(
                product,
                shift
            )

            output[
                out_pos,
                channel
            ] = saturate(
                value,
                output_bits
            )

    return output


def depthwise_same(
    x,
    kernel,
    multiplier,
    shift,
    output_bits,
    dilation=1
):
    return depthwise_conv1d_same(
        x=x,
        kernel=kernel,
        multiplier=multiplier,
        shift=shift,
        output_bits=output_bits,
        dilation=dilation
    )


# ============================================================
# SEPARABLE CONV - LEGACY VERSION
# ============================================================

def sepconv_preact(
    x,
    cfg,
    depthwise,
    pointwise,
    bias
):
    dw_req = cfg[
        "depthwise_requant"
    ]

    pw_req = cfg[
        "acc_to_preact"
    ]

    bits = int(
        cfg[
            "bits"
        ]
    )

    dw = depthwise_same(
        x=x,
        kernel=depthwise,
        multiplier=int(
            dw_req[
                "multiplier"
            ]
        ),
        shift=int(
            dw_req[
                "shift"
            ]
        ),
        output_bits=bits,
        dilation=int(
            cfg[
                "dilation"
            ]
        )
    )

    pw = conv1d_same(
        x=dw,
        kernel=pointwise,
        bias=bias,
        multiplier=int(
            pw_req[
                "multiplier"
            ]
        ),
        shift=int(
            pw_req[
                "shift"
            ]
        ),
        output_bits=int(
            cfg[
                "output_bits"
            ]
        ),
        dilation=1
    )

    return dw, pw


# ============================================================
# SEPARABLE CONV - MIXED PRECISION VERSION
# ============================================================

def sepconv_preact_mixed(
    x,
    cfg,
    depthwise,
    pointwise,
    bias,
    preact_bits
):
    dw_req = cfg[
        "depthwise_requant"
    ]

    pw_req = cfg[
        "acc_to_preact"
    ]

    layer_bits = int(
        cfg[
            "bits"
        ]
    )

    # --------------------------------------------------------
    # Depthwise
    #
    # INT8 layer:
    #   A8 x W8 -> A8
    #
    # INT4 layer:
    #   A8 x W4 -> A4
    # --------------------------------------------------------

    dw = depthwise_same(
        x=x,
        kernel=depthwise,
        multiplier=int(
            dw_req[
                "multiplier"
            ]
        ),
        shift=int(
            dw_req[
                "shift"
            ]
        ),
        output_bits=layer_bits,
        dilation=int(
            cfg[
                "dilation"
            ]
        )
    )

    # --------------------------------------------------------
    # Pointwise
    #
    # INT4 branch vẫn tạo BN/preactivation ở INT8.
    # preact_bits lấy từ tensor scale của BN.
    # --------------------------------------------------------

    pw = conv1d_same(
        x=dw,
        kernel=pointwise,
        bias=bias,
        multiplier=int(
            pw_req[
                "multiplier"
            ]
        ),
        shift=int(
            pw_req[
                "shift"
            ]
        ),
        output_bits=int(
            preact_bits
        ),
        dilation=1
    )

    return dw, pw


# ============================================================
# MAX POOLING
# ============================================================

def maxpool1d_same(
    x,
    pool_size=2,
    stride=2
):
    x = np.asarray(
        x
    )

    length, channels = (
        x.shape
    )

    output_len = (
        length
        + stride - 1
    ) // stride

    output = np.empty(
        (
            output_len,
            channels
        ),
        dtype=x.dtype
    )

    for out_pos in range(
        output_len
    ):
        start = (
            out_pos
            * stride
        )

        end = min(
            start
            + pool_size,
            length
        )

        output[
            out_pos
        ] = np.max(
            x[
                start:end
            ],
            axis=0
        )

    return output


def maxpool1d_same3_s1(
    x
):
    x = np.asarray(
        x
    )

    if x.ndim != 2:
        raise ValueError(
            "Input pooling phải có "
            f"shape [L,C], nhận {x.shape}"
        )

    length, channels = (
        x.shape
    )

    output = np.empty(
        (
            length,
            channels
        ),
        dtype=x.dtype
    )

    for t in range(
        length
    ):
        start = max(
            0,
            t - 1
        )

        end = min(
            length,
            t + 2
        )

        output[t] = np.max(
            x[
                start:end
            ],
            axis=0
        )

    return output


def maxpool1d_2_s2(
    x
):
    x = np.asarray(
        x
    )

    if x.ndim != 2:
        raise ValueError(
            "Input pooling phải có "
            f"shape [L,C], nhận {x.shape}"
        )

    length, channels = (
        x.shape
    )

    if length % 2 != 0:
        raise ValueError(
            "Length phải chẵn cho "
            "MaxPool size=2 stride=2."
        )

    output_len = (
        length // 2
    )

    output = np.empty(
        (
            output_len,
            channels
        ),
        dtype=x.dtype
    )

    for i in range(
        output_len
    ):
        output[i] = np.maximum(
            x[
                2 * i
            ],
            x[
                2 * i + 1
            ]
        )

    return output


# ============================================================
# RESIDUAL ADD
# ============================================================

def residual_add_scaled(
    main_q,
    main_scale,
    shortcut_q,
    shortcut_scale,
    output_scale,
    output_bits
):
    main_q = np.asarray(
        main_q,
        dtype=np.int8
    )

    shortcut_q = np.asarray(
        shortcut_q,
        dtype=np.int8
    )

    if (
        main_q.shape
        != shortcut_q.shape
    ):
        raise ValueError(
            "Residual shape mismatch: "
            f"{main_q.shape} "
            f"vs "
            f"{shortcut_q.shape}"
        )

    main_aligned = (
        requantize_tensor(
            q=main_q,
            input_scale=main_scale,
            output_scale=output_scale,
            bits=output_bits
        )
    ).astype(
        np.int16
    )

    shortcut_aligned = (
        requantize_tensor(
            q=shortcut_q,
            input_scale=shortcut_scale,
            output_scale=output_scale,
            bits=output_bits
        )
    ).astype(
        np.int16
    )

    summed = (
        main_aligned
        + shortcut_aligned
    )

    return clip_tensor(
        summed,
        output_bits
    )


# ============================================================
# DENSE
# ============================================================

def dense_integer(
    x,
    kernel,
    bias,
    multiplier,
    shift,
    output_bits
):
    x = np.asarray(
        x,
        dtype=np.int8
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int8
    )

    bias = np.asarray(
        bias,
        dtype=np.int32
    )

    if x.ndim != 1:
        raise ValueError(
            "Dense input phải là vector 1D, "
            f"nhận {x.shape}"
        )

    if kernel.ndim != 2:
        raise ValueError(
            "Dense kernel phải có "
            f"shape [Cin,Cout], nhận {kernel.shape}"
        )

    cin, cout = (
        kernel.shape
    )

    if x.shape[0] != cin:
        raise ValueError(
            "Dense Cin mismatch: "
            f"input={x.shape[0]}, "
            f"kernel={cin}"
        )

    if bias.shape != (
        cout,
    ):
        raise ValueError(
            "Dense bias mismatch: "
            f"{bias.shape}, Cout={cout}"
        )

    output = np.zeros(
        cout,
        dtype=np.int8
    )

    for out_ch in range(
        cout
    ):
        acc = int(
            bias[
                out_ch
            ]
        )

        for in_ch in range(
            cin
        ):
            acc += (
                int(
                    x[
                        in_ch
                    ]
                )
                *
                int(
                    kernel[
                        in_ch,
                        out_ch
                    ]
                )
            )

        product = (
            acc
            * int(multiplier)
        )

        value = round_shift_signed(
            product,
            shift
        )

        output[
            out_ch
        ] = saturate(
            value,
            output_bits
        )

    return output


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

def global_average_requant(
    x,
    input_scale,
    output_scale,
    output_bits
):
    x = np.asarray(
        x,
        dtype=np.int8
    )

    if x.ndim != 2:
        raise ValueError(
            "GAP input phải [L,C], "
            f"nhận {x.shape}"
        )

    x_float = dequantize(
        x,
        input_scale
    )

    avg_float = np.mean(
        x_float,
        axis=0
    )

    return quantize(
        avg_float,
        output_scale,
        output_bits
    )


# ============================================================
# GLOBAL MAX POOLING
# ============================================================

def global_max_requant(
    x,
    input_scale,
    output_scale,
    output_bits
):
    x = np.asarray(
        x,
        dtype=np.int8
    )

    if x.ndim != 2:
        raise ValueError(
            "GMP input phải [L,C], "
            f"nhận {x.shape}"
        )

    maximum = np.max(
        x,
        axis=0
    )

    return requantize_tensor(
        q=maximum,
        input_scale=input_scale,
        output_scale=output_scale,
        bits=output_bits
    )
def dense_logits_int32(
    x,
    kernel,
    bias
):
    x = np.asarray(
        x,
        dtype=np.int8
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int8
    )

    bias = np.asarray(
        bias,
        dtype=np.int32
    )

    if x.ndim != 1:
        raise ValueError(
            f"Logit Dense input phải 1D, nhận {x.shape}"
        )

    if kernel.ndim != 2:
        raise ValueError(
            f"Logit kernel phải [Cin,Cout], nhận {kernel.shape}"
        )

    cin, cout = kernel.shape

    if x.shape[0] != cin:
        raise ValueError(
            f"Cin mismatch: {x.shape[0]} != {cin}"
        )

    if bias.shape != (cout,):
        raise ValueError(
            f"Bias mismatch: {bias.shape}"
        )

    output = np.zeros(
        cout,
        dtype=np.int32
    )

    for oc in range(cout):

        acc = int(
            bias[oc]
        )

        for ic in range(cin):

            acc += (
                int(x[ic])
                *
                int(
                    kernel[
                        ic,
                        oc
                    ]
                )
            )

        output[oc] = acc

    return output


def softmax_float(
    logits
):
    logits = np.asarray(
        logits,
        dtype=np.float64
    )

    shifted = (
        logits
        - np.max(logits)
    )

    exp_values = np.exp(
        shifted
    )

    return (
        exp_values
        / np.sum(exp_values)
    )