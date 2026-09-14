import json
from pathlib import Path

import numpy as np

from ecg_fpga_driver import ECGFPGAAccelerator


BASE = Path(
    "/home/xilinx/didong/ecg_fpga"
)

TEST_DIR = (
    BASE
    / "real_sepconv_test"
)

BIT_FILE = (
    BASE
    / "overlay"
    / "ecg_accelerator.bit"
)


print("========================================")
print("REAL BLOCK1_B1 SEPCONV FPGA TEST")
print("========================================")


# =========================================================
# Load package
# =========================================================

x = np.load(
    TEST_DIR / "input_int8.npy"
).astype(np.int8)

depthwise = np.load(
    TEST_DIR / "depthwise_int8.npy"
).astype(np.int8)

pointwise = np.load(
    TEST_DIR / "pointwise_int8.npy"
).astype(np.int8)

bias = np.load(
    TEST_DIR / "bias_int32.npy"
).astype(np.int32)

expected_dw = np.load(
    TEST_DIR / "expected_depthwise.npy"
).astype(np.int8)

expected_pw = np.load(
    TEST_DIR / "expected_pointwise.npy"
).astype(np.int8)


with open(
    TEST_DIR / "config.json",
    "r"
) as f:
    cfg = json.load(f)


print()
print("Layer:", cfg["layer"])
print("Bits :", cfg["bits"])

print()
print("Input shape:")
print(x.shape)

print("Depthwise shape:")
print(depthwise.shape)

print("Pointwise shape:")
print(pointwise.shape)

print("Bias shape:")
print(bias.shape)

print("Expected DW:")
print(expected_dw.shape)

print("Expected PW:")
print(expected_pw.shape)


# =========================================================
# Load FPGA overlay
# =========================================================

fpga = ECGFPGAAccelerator(
    str(BIT_FILE)
)


# =========================================================
# DEPTHWISE
# =========================================================

print()
print("========================================")
print("DEPTHWISE FPGA")
print("========================================")


# Depthwise của SeparableConv không có bias riêng.
# Bias đã được fold vào pointwise / BN stage.
zero_bias = np.zeros(
    x.shape[1],
    dtype=np.int32
)


actual_dw = fpga.run_layer(
    x=x,

    weights=depthwise,

    bias=zero_bias,

    multiplier=int(
        cfg["depthwise_multiplier"]
    ),

    shift=int(
        cfg["depthwise_shift"]
    ),

    kernel_size=int(
        cfg["kernel_size"]
    ),

    dilation=int(
        cfg["dilation"]
    ),

    same=bool(
        cfg["same"]
    ),

    depthwise=True,

    int4=False,

    output_int4=False
)


print()
print(
    "FPGA DW output shape:",
    actual_dw.shape
)


if actual_dw.shape != expected_dw.shape:
    print(
        "FAIL: Depthwise shape mismatch"
    )
    print(
        "FPGA    :",
        actual_dw.shape
    )
    print(
        "Expected:",
        expected_dw.shape
    )
    raise SystemExit(1)


dw_diff = (
    actual_dw.astype(np.int16)
    -
    expected_dw.astype(np.int16)
)


dw_errors = int(
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
    dw_errors
)

print(
    "Max abs error:",
    dw_max_error
)


if dw_errors != 0:

    print()
    print(
        "FAIL: DEPTHWISE MISMATCH"
    )

    bad = np.argwhere(
        dw_diff != 0
    )

    print()
    print(
        "First mismatches:"
    )

    for idx in bad[:20]:

        i = tuple(idx)

        print(
            i,
            "FPGA=",
            int(actual_dw[i]),
            "EXPECTED=",
            int(expected_dw[i]),
            "DIFF=",
            int(dw_diff[i])
        )

    np.save(
        TEST_DIR
        / "actual_depthwise_fpga.npy",
        actual_dw
    )

    raise SystemExit(1)


print()
print(
    "PASS: DEPTHWISE IS BIT-EXACT"
)


np.save(
    TEST_DIR
    / "actual_depthwise_fpga.npy",
    actual_dw
)


# =========================================================
# POINTWISE
# =========================================================

print()
print("========================================")
print("POINTWISE FPGA")
print("========================================")


actual_pw = fpga.run_layer(
    x=actual_dw,

    weights=pointwise,

    bias=bias,

    multiplier=int(
        cfg["pointwise_multiplier"]
    ),

    shift=int(
        cfg["pointwise_shift"]
    ),

    kernel_size=1,

    dilation=1,

    same=True,

    depthwise=False,

    int4=False,

    output_int4=False
)


print()
print(
    "FPGA PW output shape:",
    actual_pw.shape
)


if actual_pw.shape != expected_pw.shape:
    print(
        "FAIL: Pointwise shape mismatch"
    )
    print(
        "FPGA    :",
        actual_pw.shape
    )
    print(
        "Expected:",
        expected_pw.shape
    )
    raise SystemExit(1)


pw_diff = (
    actual_pw.astype(np.int16)
    -
    expected_pw.astype(np.int16)
)


pw_errors = int(
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
    pw_errors
)

print(
    "Max abs error:",
    pw_max_error
)


if pw_errors != 0:

    print()
    print(
        "FAIL: POINTWISE MISMATCH"
    )

    bad = np.argwhere(
        pw_diff != 0
    )

    print()
    print(
        "First mismatches:"
    )

    for idx in bad[:20]:

        i = tuple(idx)

        print(
            i,
            "FPGA=",
            int(actual_pw[i]),
            "EXPECTED=",
            int(expected_pw[i]),
            "DIFF=",
            int(pw_diff[i])
        )

    np.save(
        TEST_DIR
        / "actual_pointwise_fpga.npy",
        actual_pw
    )

    raise SystemExit(1)


np.save(
    TEST_DIR
    / "actual_pointwise_fpga.npy",
    actual_pw
)


print()
print(
    "PASS: POINTWISE IS BIT-EXACT"
)


print()
print("========================================")
print("PASS: REAL SEPARABLE CONV IS BIT-EXACT")
print("========================================")