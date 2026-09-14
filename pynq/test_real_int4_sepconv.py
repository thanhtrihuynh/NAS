import json
from pathlib import Path

import numpy as np

from ecg_fpga_driver import (
    ECGFPGAAccelerator
)


BASE = Path(
    "/home/xilinx/didong/ecg_fpga"
)

TEST_DIR = (
    BASE
    / "real_int4_sepconv_test"
)

BIT_FILE = (
    BASE
    / "overlay"
    / "ecg_accelerator.bit"
)


print(
    "========================================"
)
print(
    "REAL BLOCK2_B3 INT4 FPGA TEST"
)
print(
    "========================================"
)


with open(
    TEST_DIR
    / "config.json",
    "r"
) as f:
    cfg = json.load(f)


x = np.load(
    TEST_DIR
    / "input_pool1_int8.npy"
).astype(
    np.int8
)

depthwise = np.load(
    TEST_DIR
    / "depthwise_int4.npy"
).astype(
    np.int8
)

pointwise = np.load(
    TEST_DIR
    / "pointwise_int4.npy"
).astype(
    np.int8
)

bias = np.load(
    TEST_DIR
    / "bias_int32.npy"
).astype(
    np.int32
)

expected_dw = np.load(
    TEST_DIR
    / "expected_depthwise_int4.npy"
).astype(
    np.int8
)

expected_pw = np.load(
    TEST_DIR
    / "expected_pointwise_int8.npy"
).astype(
    np.int8
)


print()
print(
    "Input:",
    x.shape,
    "range=",
    int(x.min()),
    "...",
    int(x.max())
)

print(
    "DW weights:",
    depthwise.shape,
    "range=",
    int(depthwise.min()),
    "...",
    int(depthwise.max())
)

print(
    "PW weights:",
    pointwise.shape,
    "range=",
    int(pointwise.min()),
    "...",
    int(pointwise.max())
)


fpga = ECGFPGAAccelerator(
    str(BIT_FILE)
)


# ========================================================
# DEPTHWISE
#
# Input activations vẫn là INT8.
# Weights có giá trị INT4 nhưng hiện được lưu trong
# container int8 của accelerator.
#
# Vì vậy KHÔNG bật int4 input mode ở stage này.
#
# Output phải saturate INT4.
# ========================================================

print()
print(
    "========================================"
)
print(
    "DEPTHWISE A8 x W4 -> A4"
)
print(
    "========================================"
)


zero_bias = np.zeros(
    8,
    dtype=np.int32
)


actual_dw = fpga.run_layer(
    x=x,

    weights=depthwise,

    bias=zero_bias,

    multiplier=int(
        cfg[
            "depthwise_multiplier"
        ]
    ),

    shift=int(
        cfg[
            "depthwise_shift"
        ]
    ),

    kernel_size=9,

    dilation=2,

    same=True,

    depthwise=True,

    # Input vẫn INT8.
    # Weight chỉ có giá trị trong miền INT4.
    int4=False,

    # Depthwise output phải A4.
    output_int4=True
)


print()
print(
    "FPGA DW shape:",
    actual_dw.shape
)

print(
    "FPGA DW range:",
    int(actual_dw.min()),
    "...",
    int(actual_dw.max())
)


dw_diff = (
    actual_dw.astype(
        np.int16
    )
    -
    expected_dw.astype(
        np.int16
    )
)


dw_mismatch = int(
    np.count_nonzero(
        dw_diff
    )
)


dw_max_error = int(
    np.max(
        np.abs(
            dw_diff
        )
    )
)


print(
    "Total elements:",
    actual_dw.size
)

print(
    "Mismatch count:",
    dw_mismatch
)

print(
    "Max abs error:",
    dw_max_error
)


if dw_mismatch != 0:

    bad = np.argwhere(
        dw_diff != 0
    )

    print()
    print(
        "First mismatches:"
    )

    for idx in bad[:20]:

        idx = tuple(idx)

        print(
            idx,
            "FPGA=",
            int(
                actual_dw[idx]
            ),
            "EXPECTED=",
            int(
                expected_dw[idx]
            ),
            "DIFF=",
            int(
                dw_diff[idx]
            )
        )

    raise SystemExit(1)


print()
print(
    "PASS: DEPTHWISE A8xW4 -> A4 BIT-EXACT"
)


# ========================================================
# POINTWISE
#
# Input actual_dw giờ đã nằm trong [-7, 7].
# Weight cũng INT4.
#
# Vì vậy đây là A4 x W4.
#
# Nhưng output của Pointwise là block2_b3_bn /
# preactivation, nên output phải INT8.
# ========================================================

print()
print(
    "========================================"
)
print(
    "POINTWISE A4 x W4 -> A8"
)
print(
    "========================================"
)


actual_pw = fpga.run_layer(
    x=actual_dw,

    weights=pointwise,

    bias=bias,

    multiplier=int(
        cfg[
            "pointwise_multiplier"
        ]
    ),

    shift=int(
        cfg[
            "pointwise_shift"
        ]
    ),

    kernel_size=1,

    dilation=1,

    same=True,

    depthwise=False,

    # Cả input và weights đều nằm trong miền INT4.
    int4=True,

    # Pointwise preactivation vẫn là INT8.
    output_int4=False
)


print()
print(
    "FPGA PW shape:",
    actual_pw.shape
)

print(
    "FPGA PW range:",
    int(actual_pw.min()),
    "...",
    int(actual_pw.max())
)


pw_diff = (
    actual_pw.astype(
        np.int16
    )
    -
    expected_pw.astype(
        np.int16
    )
)


pw_mismatch = int(
    np.count_nonzero(
        pw_diff
    )
)


pw_max_error = int(
    np.max(
        np.abs(
            pw_diff
        )
    )
)


print(
    "Total elements:",
    actual_pw.size
)

print(
    "Mismatch count:",
    pw_mismatch
)

print(
    "Max abs error:",
    pw_max_error
)


if pw_mismatch != 0:

    bad = np.argwhere(
        pw_diff != 0
    )

    print()
    print(
        "First mismatches:"
    )

    for idx in bad[:20]:

        idx = tuple(idx)

        print(
            idx,
            "FPGA=",
            int(
                actual_pw[idx]
            ),
            "EXPECTED=",
            int(
                expected_pw[idx]
            ),
            "DIFF=",
            int(
                pw_diff[idx]
            )
        )

    raise SystemExit(1)


print()
print(
    "PASS: POINTWISE A4xW4 -> A8 BIT-EXACT"
)


print()
print(
    "========================================"
)
print(
    "PASS: REAL MIXED-PRECISION SEPCONV"
)
print(
    "========================================"
)