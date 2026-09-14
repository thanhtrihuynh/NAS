import numpy as np


def max_pool1d_integer(
    x,
    pool_size=2,
    stride=None,
    padding="valid"
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    if stride is None:
        stride = pool_size

    pool_size = int(
        pool_size
    )

    stride = int(
        stride
    )

    batch, length, channels = (
        x.shape
    )

    padding = padding.lower()

    if padding == "valid":
        output_length = (
            (
                length
                - pool_size
            )
            // stride
            + 1
        )

        x_pad = x

    elif padding == "same":
        output_length = int(
            np.ceil(
                length
                / stride
            )
        )

        total_padding = max(
            (
                output_length - 1
            )
            * stride
            + pool_size
            - length,
            0
        )

        pad_left = (
            total_padding // 2
        )

        pad_right = (
            total_padding
            - pad_left
        )

        minimum = np.iinfo(
            np.int64
        ).min

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
            mode="constant",
            constant_values=minimum
        )

    else:
        raise ValueError(
            f"Padding chưa hỗ trợ: "
            f"{padding}"
        )

    output = np.empty(
        (
            batch,
            output_length,
            channels
        ),
        dtype=np.int64
    )

    for index in range(
        output_length
    ):
        start = (
            index
            * stride
        )

        end = (
            start
            + pool_size
        )

        window = (
            x_pad[
                :,
                start:end,
                :
            ]
        )

        output[
            :,
            index,
            :
        ] = np.max(
            window,
            axis=1
        )

    return output


def global_max_pool1d_integer(
    x
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    return np.max(
        x,
        axis=1
    )


def round_divide_signed(
    values,
    divisor
):
    values = np.asarray(
        values,
        dtype=np.int64
    )

    divisor = int(
        divisor
    )

    if divisor <= 0:
        raise ValueError(
            "divisor phải > 0"
        )

    positive = (
        values >= 0
    )

    result = np.empty_like(
        values
    )

    half = (
        divisor // 2
    )

    result[positive] = (
        values[positive]
        + half
    ) // divisor

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
            + half
        )
        // divisor
    )

    return result


def global_average_pool1d_integer(
    x
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    length = (
        x.shape[1]
    )

    summed = np.sum(
        x,
        axis=1,
        dtype=np.int64
    )

    return round_divide_signed(
        summed,
        length
    )