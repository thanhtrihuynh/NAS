import numpy as np

from quantization.fixed_point import (
    qrange,
    multiplier_shift,
    rounding_right_shift
)


def rescale_integer_no_clip(
    x,
    source_scale,
    target_scale
):
    x = np.asarray(
        x,
        dtype=np.int64
    )

    source_scale = float(
        source_scale
    )

    target_scale = float(
        target_scale
    )

    if source_scale <= 0:
        raise ValueError(
            "source_scale phải > 0"
        )

    if target_scale <= 0:
        raise ValueError(
            "target_scale phải > 0"
        )

    if abs(
        source_scale
        - target_scale
    ) < 1e-15:
        return x

    ratio = (
        source_scale
        / target_scale
    )

    fixed = multiplier_shift(
        ratio
    )

    product = (
        x
        * int(
            fixed[
                "multiplier"
            ]
        )
    )

    result = (
        rounding_right_shift(
            product,
            int(
                fixed[
                    "shift"
                ]
            )
        )
    )

    return result.astype(
        np.int64
    )


def align_scale(
    x,
    source_scale,
    target_scale,
    target_bits
):
    result = (
        rescale_integer_no_clip(
            x,
            source_scale,
            target_scale
        )
    )

    qmin, qmax = qrange(
        target_bits
    )

    result = np.clip(
        result,
        qmin,
        qmax
    )

    return result.astype(
        np.int8
    )


def concat_integer(
    tensors,
    scales,
    target_scale,
    target_bits,
    axis=-1
):
    if len(tensors) != len(
        scales
    ):
        raise ValueError(
            "Số tensor và scale không khớp."
        )

    aligned = []

    for tensor, scale in zip(
        tensors,
        scales
    ):
        aligned.append(
            align_scale(
                tensor,
                scale,
                target_scale,
                target_bits
            )
        )

    return np.concatenate(
        aligned,
        axis=axis
    )


def residual_add_integer(
    main,
    main_scale,
    shortcut,
    shortcut_scale,
    output_scale,
    output_bits
):
    main_scaled = (
        rescale_integer_no_clip(
            main,
            main_scale,
            output_scale
        )
    )

    shortcut_scaled = (
        rescale_integer_no_clip(
            shortcut,
            shortcut_scale,
            output_scale
        )
    )

    result = (
        main_scaled
        + shortcut_scaled
    )

    qmin, qmax = qrange(
        output_bits
    )

    result = np.clip(
        result,
        qmin,
        qmax
    )

    return result.astype(
        np.int8
    )