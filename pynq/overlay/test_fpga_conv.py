import numpy as np

from ecg_fpga_driver import (
    ECGFPGAAccelerator
)


BIT_FILE = (
    "overlay/ecg_accelerator.bit"
)


fpga = ECGFPGAAccelerator(
    BIT_FILE
)


# =========================================================
# Input
# =========================================================

x = np.array(
    [
        [1],
        [2],
        [3],
        [4]
    ],
    dtype=np.int8
)


# =========================================================
# K = 3
# Cin = 1
# Cout = 1
#
# Kernel:
# [1, 2, 1]
# =========================================================

weights = np.array(
    [
        [[1]],
        [[2]],
        [[1]]
    ],
    dtype=np.int8
)


bias = np.array(
    [0],
    dtype=np.int32
)


# =========================================================
# multiplier = 2^30
# shift      = 30
#
# ratio = 1
# =========================================================

result = fpga.run_layer(
    x=x,

    weights=weights,

    bias=bias,

    multiplier=1073741824,

    shift=30,

    kernel_size=3,

    dilation=1,

    same=True,

    depthwise=False,

    int4=False,

    output_int4=False
)


expected = np.array(
    [
        [4],
        [8],
        [12],
        [11]
    ],
    dtype=np.int8
)


print()
print(
    "FPGA output:"
)

print(
    result.reshape(-1)
)


print()
print(
    "Expected:"
)

print(
    expected.reshape(-1)
)


print()


if np.array_equal(
    result,
    expected
):

    print(
        "PASS: PYNQ -> DMA -> FPGA "
        "-> DMA -> PYNQ"
    )

else:

    print(
        "FAIL"
    )

    print(
        "Difference:"
    )

    print(
        result.astype(
            np.int16
        )
        -
        expected.astype(
            np.int16
        )
    )