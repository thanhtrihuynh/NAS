import numpy as np

from quantization.fixed_point import (
    qrange,
    multiplier_shift,
    rounding_right_shift
)


def quantize_tensor(
    x,
    scale,
    bits
):
    x = np.asarray(
        x,
        dtype=np.float32
    )

    scale = float(
        scale
    )

    if scale <= 0:
        raise ValueError(
            "scale phải > 0"
        )

    qmin, qmax = qrange(
        bits
    )

    q = np.round(
        x / scale
    )

    q = np.clip(
        q,
        qmin,
        qmax
    )

    return q.astype(
        np.int8
    )


def dequantize_tensor(
    q,
    scale
):
    return (
        np.asarray(
            q,
            dtype=np.float32
        )
        * float(scale)
    )


def compute_same_padding(
    input_length,
    kernel_size,
    stride,
    dilation
):
    effective_kernel = (
        dilation
        * (
            kernel_size - 1
        )
        + 1
    )

    output_length = int(
        np.ceil(
            input_length
            / stride
        )
    )

    total_padding = max(
        (
            output_length - 1
        )
        * stride
        + effective_kernel
        - input_length,
        0
    )

    pad_left = (
        total_padding // 2
    )

    pad_right = (
        total_padding
        - pad_left
    )

    return (
        pad_left,
        pad_right,
        output_length
    )


def integer_conv1d_acc(
    x,
    kernel,
    bias,
    stride=1,
    dilation=1,
    padding="same"
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int64
    )

    bias = np.asarray(
        bias,
        dtype=np.int64
    )

    (
        batch,
        length,
        cin
    ) = x.shape

    (
        kernel_size,
        kernel_cin,
        cout
    ) = kernel.shape

    if cin != kernel_cin:
        raise ValueError(
            f"Cin mismatch: "
            f"{cin} != {kernel_cin}"
        )

    stride = int(
        stride
    )

    dilation = int(
        dilation
    )

    effective_kernel = (
        dilation
        * (
            kernel_size - 1
        )
        + 1
    )

    if padding.lower() == "same":
        (
            pad_left,
            pad_right,
            output_length
        ) = compute_same_padding(
            length,
            kernel_size,
            stride,
            dilation
        )

    elif padding.lower() == "valid":
        pad_left = 0
        pad_right = 0

        output_length = (
            (
                length
                - effective_kernel
            )
            // stride
            + 1
        )

    else:
        raise ValueError(
            f"Padding chưa hỗ trợ: "
            f"{padding}"
        )

    x_pad = np.pad(
        x,
        (
            (0, 0),
            (
                pad_left,
                pad_right
            ),
            (0, 0)
        ),
        mode="constant"
    )

    output = np.empty(
        (
            batch,
            output_length,
            cout
        ),
        dtype=np.int64
    )

    kernel_indices = (
        np.arange(
            kernel_size
        )
        * dilation
    )

    for output_index in range(
        output_length
    ):
        start = (
            output_index
            * stride
        )

        indices = (
            start
            + kernel_indices
        )

        window = (
            x_pad[
                :,
                indices,
                :
            ]
        )

        output[
            :,
            output_index,
            :
        ] = np.tensordot(
            window,
            kernel,
            axes=(
                [1, 2],
                [0, 1]
            )
        )

    output += bias.reshape(
        1,
        1,
        -1
    )

    return output


def integer_depthwise_conv1d_acc(
    x,
    kernel,
    stride=1,
    dilation=1,
    padding="same"
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int64
    )

    (
        batch,
        length,
        cin
    ) = x.shape

    (
        kernel_size,
        kernel_cin,
        depth_multiplier
    ) = kernel.shape

    if cin != kernel_cin:
        raise ValueError(
            "Depthwise Cin mismatch: "
            f"{cin} != {kernel_cin}"
        )

    stride = int(
        stride
    )

    dilation = int(
        dilation
    )

    effective_kernel = (
        dilation
        * (
            kernel_size - 1
        )
        + 1
    )

    if padding.lower() == "same":
        (
            pad_left,
            pad_right,
            output_length
        ) = compute_same_padding(
            length,
            kernel_size,
            stride,
            dilation
        )

    elif padding.lower() == "valid":
        pad_left = 0
        pad_right = 0

        output_length = (
            (
                length
                - effective_kernel
            )
            // stride
            + 1
        )

    else:
        raise ValueError(
            f"Padding chưa hỗ trợ: "
            f"{padding}"
        )

    x_pad = np.pad(
        x,
        (
            (0, 0),
            (
                pad_left,
                pad_right
            ),
            (0, 0)
        ),
        mode="constant"
    )

    output = np.empty(
        (
            batch,
            output_length,
            cin * depth_multiplier
        ),
        dtype=np.int64
    )

    kernel_indices = (
        np.arange(
            kernel_size
        )
        * dilation
    )

    for output_index in range(
        output_length
    ):
        start = (
            output_index
            * stride
        )

        indices = (
            start
            + kernel_indices
        )

        window = (
            x_pad[
                :,
                indices,
                :
            ]
        )

        result = np.einsum(
            "bkc,kcm->bcm",
            window,
            kernel,
            dtype=np.int64
        )

        output[
            :,
            output_index,
            :
        ] = result.reshape(
            batch,
            cin * depth_multiplier
        )

    return output


def integer_pointwise_conv1d_acc(
    x,
    kernel,
    bias
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int64
    )

    bias = np.asarray(
        bias,
        dtype=np.int64
    )

    if kernel.shape[0] != 1:
        raise ValueError(
            "Pointwise kernel phải có K=1"
        )

    pointwise_kernel = (
        kernel[0]
    )

    output = np.tensordot(
        x,
        pointwise_kernel,
        axes=(
            [2],
            [0]
        )
    )

    output += bias.reshape(
        1,
        1,
        -1
    )

    return output.astype(
        np.int64
    )


def integer_dense_acc(
    x,
    kernel,
    bias
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    kernel = np.asarray(
        kernel,
        dtype=np.int64
    )

    bias = np.asarray(
        bias,
        dtype=np.int64
    )

    output = (
        np.matmul(
            x,
            kernel
        )
        + bias
    )

    return output.astype(
        np.int64
    )


def requantize_from_scale(
    accumulator,
    source_scale,
    target_scale,
    target_bits
):
    source_scale = float(
        source_scale
    )

    target_scale = float(
        target_scale
    )

    if target_scale <= 0:
        raise ValueError(
            "target_scale phải > 0"
        )

    ratio = (
        source_scale
        / target_scale
    )

    fixed = multiplier_shift(
        ratio
    )

    product = (
        np.asarray(
            accumulator,
            dtype=np.int64
        )
        * int(
            fixed[
                "multiplier"
            ]
        )
    )

    shifted = (
        rounding_right_shift(
            product,
            int(
                fixed[
                    "shift"
                ]
            )
        )
    )

    qmin, qmax = qrange(
        target_bits
    )

    shifted = np.clip(
        shifted,
        qmin,
        qmax
    )

    return shifted.astype(
        np.int8
    )