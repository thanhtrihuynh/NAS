import math
import numpy as np

from quantization.fixed_point import (
    qrange
)


def gelu_float(
    x
):
    x = np.asarray(
        x,
        dtype=np.float64
    )

    erf_values = np.vectorize(
        math.erf
    )(
        x
        / math.sqrt(2.0)
    )

    return (
        0.5
        * x
        * (
            1.0
            + erf_values
        )
    )


def build_gelu_lut(
    input_scale,
    input_bits,
    output_scale,
    output_bits
):
    input_scale = float(
        input_scale
    )

    output_scale = float(
        output_scale
    )

    if input_scale <= 0:
        raise ValueError(
            "input_scale phải > 0"
        )

    if output_scale <= 0:
        raise ValueError(
            "output_scale phải > 0"
        )

    input_min, input_max = (
        qrange(
            input_bits
        )
    )

    output_min, output_max = (
        qrange(
            output_bits
        )
    )

    integer_inputs = np.arange(
        input_min,
        input_max + 1,
        dtype=np.int32
    )

    float_inputs = (
        integer_inputs.astype(
            np.float64
        )
        * input_scale
    )

    float_outputs = (
        gelu_float(
            float_inputs
        )
    )

    integer_outputs = np.round(
        float_outputs
        / output_scale
    )

    integer_outputs = np.clip(
        integer_outputs,
        output_min,
        output_max
    )

    return integer_outputs.astype(
        np.int8
    )


def gelu_integer_lut(
    x,
    input_scale,
    input_bits,
    output_scale,
    output_bits
):
    x = np.asarray(
        x
    )

    input_min, input_max = (
        qrange(
            input_bits
        )
    )

    table = build_gelu_lut(
        input_scale,
        input_bits,
        output_scale,
        output_bits
    )

    clipped = np.clip(
        x.astype(
            np.int64
        ),
        input_min,
        input_max
    )

    indices = (
        clipped
        - input_min
    )

    return table[
        indices
    ]