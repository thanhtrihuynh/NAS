import numpy as np


INT32_MIN = -(2 ** 31)
INT32_MAX = (2 ** 31) - 1


def qrange(bits):
    bits = int(bits)

    if bits == 8:
        return -127, 127

    if bits == 4:
        return -7, 7

    raise ValueError(
        f"Chỉ hỗ trợ INT4 hoặc INT8, nhận bits={bits}"
    )


def symmetric_scale(values, bits):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    _, qmax = qrange(bits)

    max_abs = float(
        np.max(
            np.abs(values)
        )
    )

    if max_abs < 1e-12:
        return 1.0

    return max_abs / float(qmax)


def quantize_symmetric(
    values,
    bits,
    scale=None
):
    values = np.asarray(
        values,
        dtype=np.float32
    )

    qmin, qmax = qrange(bits)

    if scale is None:
        scale = symmetric_scale(
            values,
            bits
        )

    scale = float(scale)

    if scale <= 0:
        raise ValueError(
            "Quantization scale phải > 0."
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
        q.astype(np.int8),
        scale
    )


def dequantize_symmetric(
    values,
    scale
):
    return (
        np.asarray(
            values,
            dtype=np.float32
        )
        * float(scale)
    )


def quantize_bias_int32(
    bias_float,
    bias_scale
):
    bias_scale = float(
        bias_scale
    )

    if bias_scale <= 0:
        raise ValueError(
            "bias_scale phải > 0."
        )

    bias_float = np.asarray(
        bias_float,
        dtype=np.float64
    )

    q = np.round(
        bias_float / bias_scale
    )

    q = np.clip(
        q,
        INT32_MIN,
        INT32_MAX
    )

    return q.astype(
        np.int32
    )


def multiplier_shift(
    real_multiplier
):
    """
    Tìm M và shift sao cho:

        real_multiplier ≈ M / 2^shift

    M dùng signed INT32.
    """

    real_multiplier = float(
        real_multiplier
    )

    if real_multiplier < 0:
        raise ValueError(
            "real_multiplier phải >= 0."
        )

    if real_multiplier == 0.0:
        return {
            "multiplier": 0,
            "shift": 0,
            "real_multiplier": 0.0,
            "approx_multiplier": 0.0,
            "error": 0.0
        }

    best = None

    for shift in range(
        31,
        -1,
        -1
    ):
        multiplier = int(
            round(
                real_multiplier
                * (2 ** shift)
            )
        )

        if (
            0 <= multiplier <= INT32_MAX
        ):
            approximation = (
                multiplier
                / float(
                    2 ** shift
                )
            )

            best = {
                "multiplier":
                    multiplier,

                "shift":
                    shift,

                "real_multiplier":
                    real_multiplier,

                "approx_multiplier":
                    approximation,

                "error":
                    abs(
                        real_multiplier
                        - approximation
                    )
            }

            break

    if best is None:
        raise OverflowError(
            "Không biểu diễn được multiplier: "
            f"{real_multiplier}"
        )

    return best


def rounding_right_shift(
    values,
    shift
):
    """
    Signed round-to-nearest,
    ties away from zero.
    """

    values = np.asarray(
        values,
        dtype=np.int64
    )

    shift = int(shift)

    if shift < 0:
        raise ValueError(
            "shift phải >= 0."
        )

    if shift == 0:
        return values

    offset = (
        1 << (shift - 1)
    )

    positive = (
        values >= 0
    )

    result = np.empty_like(
        values,
        dtype=np.int64
    )

    result[positive] = (
        values[positive]
        + offset
    ) >> shift

    negative_abs = (
        -values[
            ~positive
        ]
    )

    result[
        ~positive
    ] = -(
        (
            negative_abs
            + offset
        )
        >> shift
    )

    return result


def requantize_integer(
    accumulator,
    multiplier,
    shift,
    output_bits
):
    accumulator = np.asarray(
        accumulator,
        dtype=np.int64
    )

    product = (
        accumulator
        * int(multiplier)
    )

    shifted = rounding_right_shift(
        product,
        shift
    )

    qmin, qmax = qrange(
        output_bits
    )

    shifted = np.clip(
        shifted,
        qmin,
        qmax
    )

    return shifted.astype(
        np.int8
    )