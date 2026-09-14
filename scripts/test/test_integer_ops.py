import numpy as np

from integer_reference.pooling import (
    max_pool1d_integer,
    global_average_pool1d_integer,
    global_max_pool1d_integer
)

from integer_reference.merge_ops import (
    concat_integer,
    residual_add_integer
)

from integer_reference.activation import (
    gelu_integer_lut
)


def main():
    print(
        "=============================="
    )
    print(
        "TEST INTEGER OPS"
    )
    print(
        "=============================="
    )

    x = np.array(
        [
            [
                [1, -2],
                [5, 3],
                [-1, 7],
                [4, 2]
            ]
        ],
        dtype=np.int8
    )

    print(
        "\nInput:"
    )

    print(x)

    pooled = max_pool1d_integer(
        x,
        pool_size=2,
        stride=2,
        padding="valid"
    )

    print(
        "\nMaxPool:"
    )

    print(pooled)

    gap = (
        global_average_pool1d_integer(
            x
        )
    )

    print(
        "\nGlobal Average Pool:"
    )

    print(gap)

    gmp = (
        global_max_pool1d_integer(
            x
        )
    )

    print(
        "\nGlobal Max Pool:"
    )

    print(gmp)

    a = np.array(
        [[[10, 20]]],
        dtype=np.int8
    )

    b = np.array(
        [[[5, -10]]],
        dtype=np.int8
    )

    added = (
        residual_add_integer(
            a,
            0.01,
            b,
            0.02,
            0.02,
            8
        )
    )

    print(
        "\nResidual Add:"
    )

    print(added)

    c1 = np.array(
        [[[10, 20]]],
        dtype=np.int8
    )

    c2 = np.array(
        [[[5, 6]]],
        dtype=np.int8
    )

    concatenated = (
        concat_integer(
            [
                c1,
                c2
            ],
            [
                0.01,
                0.02
            ],
            target_scale=0.02,
            target_bits=8,
            axis=-1
        )
    )

    print(
        "\nConcat:"
    )

    print(
        concatenated
    )

    gelu_input = np.array(
        [
            [
                -127,
                -64,
                -10,
                0,
                10,
                64,
                127
            ]
        ],
        dtype=np.int8
    )

    gelu_output = (
        gelu_integer_lut(
            gelu_input,
            input_scale=0.02,
            input_bits=8,
            output_scale=0.02,
            output_bits=8
        )
    )

    print(
        "\nGELU LUT:"
    )

    print(
        gelu_output
    )

    print()
    print(
        "INTEGER OPS TEST DONE"
    )


if __name__ == "__main__":
    main()