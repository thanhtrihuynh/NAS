import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

FINAL_DIR = (
    ROOT
    / "artifacts"
    / "final_w4a4_p99_9"
)

DEPLOY_DIR = (
    FINAL_DIR
    / "deployment_model"
)

MANIFEST_FILE = (
    DEPLOY_DIR
    / "deployment_manifest.json"
)

INPUT_DIR = (
    ROOT
    / "pynq"
    / "real_int4_test"
)

POOL1_FILE = (
    INPUT_DIR
    / "pool1_int8.npy"
)

OUTPUT_DIR = (
    ROOT
    / "pynq"
    / "real_int4_sepconv_test"
)


def round_shift_signed(
    value,
    shift
):
    value = int(value)
    shift = int(shift)

    if shift == 0:
        return value

    bias = 1 << (shift - 1)

    if value >= 0:
        return (
            value + bias
        ) >> shift

    return -(
        (
            (-value) + bias
        ) >> shift
    )


def saturate(
    value,
    bits
):
    if bits == 4:
        qmin = -7
        qmax = 7

    elif bits == 8:
        qmin = -127
        qmax = 127

    else:
        raise ValueError(
            f"Unsupported bits={bits}"
        )

    return max(
        qmin,
        min(
            qmax,
            int(value)
        )
    )


def depthwise_same(
    x,
    kernel,
    multiplier,
    shift,
    dilation,
    output_bits
):
    length, cin = x.shape

    kernel_size, kernel_cin, dm = (
        kernel.shape
    )

    if kernel_cin != cin:
        raise ValueError(
            "Depthwise Cin mismatch."
        )

    if dm != 1:
        raise ValueError(
            "Chỉ hỗ trợ depth_multiplier=1."
        )

    padding = (
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

    for t in range(length):

        for c in range(cin):

            acc = 0

            for k in range(
                kernel_size
            ):

                src = (
                    t
                    + k * dilation
                    - padding
                )

                if (
                    src < 0
                    or src >= length
                ):
                    continue

                acc += (
                    int(
                        x[src, c]
                    )
                    *
                    int(
                        kernel[
                            k,
                            c,
                            0
                        ]
                    )
                )

            product = (
                acc
                * int(multiplier)
            )

            value = (
                round_shift_signed(
                    product,
                    shift
                )
            )

            output[t, c] = saturate(
                value,
                output_bits
            )

    return output


def pointwise(
    x,
    kernel,
    bias,
    multiplier,
    shift,
    output_bits
):
    length, cin = x.shape

    if kernel.shape[0] != 1:
        raise ValueError(
            "Pointwise kernel phải K=1."
        )

    if kernel.shape[1] != cin:
        raise ValueError(
            "Pointwise Cin mismatch."
        )

    cout = kernel.shape[2]

    if bias.shape != (cout,):
        raise ValueError(
            "Bias shape mismatch."
        )

    output = np.zeros(
        (
            length,
            cout
        ),
        dtype=np.int8
    )

    for t in range(length):

        for oc in range(cout):

            acc = int(
                bias[oc]
            )

            for ic in range(cin):

                acc += (
                    int(
                        x[t, ic]
                    )
                    *
                    int(
                        kernel[
                            0,
                            ic,
                            oc
                        ]
                    )
                )

            product = (
                acc
                * int(multiplier)
            )

            value = (
                round_shift_signed(
                    product,
                    shift
                )
            )

            output[t, oc] = saturate(
                value,
                output_bits
            )

    return output


def gelu_float(x):
    x = np.asarray(
        x,
        dtype=np.float64
    )

    erf_func = np.vectorize(
        math.erf,
        otypes=[np.float64]
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


def gelu_requant_int4(
    q,
    input_scale,
    output_scale
):
    x = (
        q.astype(
            np.float64
        )
        * float(
            input_scale
        )
    )

    y = gelu_float(x)

    out = np.rint(
        y
        / float(
            output_scale
        )
    )

    out = np.clip(
        out,
        -7,
        7
    )

    return out.astype(
        np.int8
    )


def main():

    print(
        "=============================================="
    )
    print(
        "REAL BLOCK2_B3 INT4 SEPCONV GENERATOR"
    )
    print(
        "=============================================="
    )

    if not POOL1_FILE.exists():
        raise FileNotFoundError(
            POOL1_FILE
        )

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    cfg = manifest[
        "layers"
    ][
        "block2_b3_sepconv"
    ]

    # -------------------------------------------------
    # Input
    # -------------------------------------------------

    pool1 = np.load(
        POOL1_FILE
    ).astype(
        np.int8
    )

    print()
    print(
        "POOL1 input:"
    )

    print(
        "shape:",
        pool1.shape
    )

    print(
        "dtype:",
        pool1.dtype
    )

    print(
        "range:",
        int(pool1.min()),
        "...",
        int(pool1.max())
    )

    print(
        "scale:",
        cfg["input_scale"]
    )

    if pool1.shape != (
        160,
        8
    ):
        raise ValueError(
            f"pool1 shape sai: "
            f"{pool1.shape}"
        )

    # QUAN TRỌNG:
    # pool1 vẫn INT8.
    # Không clip xuống [-7, 7].

    # -------------------------------------------------
    # Weights
    # -------------------------------------------------

    depthwise = np.load(
        DEPLOY_DIR
        / cfg[
            "depthwise_file"
        ]
    ).astype(
        np.int8
    )

    pointwise_w = np.load(
        DEPLOY_DIR
        / cfg[
            "pointwise_file"
        ]
    ).astype(
        np.int8
    )

    bias = np.load(
        DEPLOY_DIR
        / cfg[
            "bias_file"
        ]
    ).astype(
        np.int32
    )

    print()
    print(
        "Depthwise:"
    )

    print(
        "shape:",
        depthwise.shape
    )

    print(
        "range:",
        int(depthwise.min()),
        "...",
        int(depthwise.max())
    )

    print()
    print(
        "Pointwise:"
    )

    print(
        "shape:",
        pointwise_w.shape
    )

    print(
        "range:",
        int(pointwise_w.min()),
        "...",
        int(pointwise_w.max())
    )

    print()
    print(
        "Bias:",
        bias.shape
    )

    if depthwise.shape != (
        9,
        8,
        1
    ):
        raise ValueError(
            "Depthwise shape sai."
        )

    if pointwise_w.shape != (
        1,
        8,
        6
    ):
        raise ValueError(
            "Pointwise shape sai."
        )

    if (
        depthwise.min() < -7
        or depthwise.max() > 7
    ):
        raise ValueError(
            "Depthwise weight "
            "không nằm trong INT4."
        )

    if (
        pointwise_w.min() < -7
        or pointwise_w.max() > 7
    ):
        raise ValueError(
            "Pointwise weight "
            "không nằm trong INT4."
        )

    # -------------------------------------------------
    # Depthwise A8 x W4 -> A4
    # -------------------------------------------------

    dw_req = cfg[
        "depthwise_requant"
    ]

    expected_dw = depthwise_same(
        x=pool1,

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

        dilation=int(
            cfg[
                "dilation"
            ]
        ),

        output_bits=4
    )

    print()
    print(
        "Depthwise expected:"
    )

    print(
        "shape:",
        expected_dw.shape
    )

    print(
        "range:",
        int(expected_dw.min()),
        "...",
        int(expected_dw.max())
    )

    # -------------------------------------------------
    # Pointwise A4 x W4 -> preactivation A8
    # -------------------------------------------------

    pw_req = cfg[
        "acc_to_preact"
    ]

    expected_pw = pointwise(
        x=expected_dw,

        kernel=pointwise_w,

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

        # RẤT QUAN TRỌNG:
        # Đây là block2_b3_bn / preactivation
        # nên vẫn phải là INT8.
        output_bits=8
    )

    print()
    print(
        "Pointwise preactivation expected:"
    )

    print(
        "shape:",
        expected_pw.shape
    )

    print(
        "range:",
        int(expected_pw.min()),
        "...",
        int(expected_pw.max())
    )

    # -------------------------------------------------
    # GELU + requant A8 -> A4
    # -------------------------------------------------

    expected_act = (
        gelu_requant_int4(
            q=expected_pw,

            input_scale=float(
                cfg[
                    "preactivation_scale"
                ]
            ),

            output_scale=float(
                cfg[
                    "output_scale"
                ]
            )
        )
    )

    print()
    print(
        "Final block2_b3_act:"
    )

    print(
        "shape:",
        expected_act.shape
    )

    print(
        "range:",
        int(expected_act.min()),
        "...",
        int(expected_act.max())
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR
        / "input_pool1_int8.npy",
        pool1
    )

    np.save(
        OUTPUT_DIR
        / "depthwise_int4.npy",
        depthwise
    )

    np.save(
        OUTPUT_DIR
        / "pointwise_int4.npy",
        pointwise_w
    )

    np.save(
        OUTPUT_DIR
        / "bias_int32.npy",
        bias
    )

    np.save(
        OUTPUT_DIR
        / "expected_depthwise_int4.npy",
        expected_dw
    )

    np.save(
        OUTPUT_DIR
        / "expected_pointwise_int8.npy",
        expected_pw
    )

    np.save(
        OUTPUT_DIR
        / "expected_activation_int4.npy",
        expected_act
    )

    config = {
        "layer":
            "block2_b3_sepconv",

        "input_shape":
            list(pool1.shape),

        "input_bits":
            8,

        "input_scale":
            float(
                cfg[
                    "input_scale"
                ]
            ),

        "weight_bits":
            4,

        "kernel_size":
            9,

        "dilation":
            2,

        "cin":
            8,

        "cout":
            6,

        "depthwise_output_bits":
            4,

        "depthwise_output_scale":
            float(
                cfg[
                    "depthwise_output_scale"
                ]
            ),

        "depthwise_multiplier":
            int(
                dw_req[
                    "multiplier"
                ]
            ),

        "depthwise_shift":
            int(
                dw_req[
                    "shift"
                ]
            ),

        "pointwise_output_bits":
            8,

        "preactivation_scale":
            float(
                cfg[
                    "preactivation_scale"
                ]
            ),

        "pointwise_multiplier":
            int(
                pw_req[
                    "multiplier"
                ]
            ),

        "pointwise_shift":
            int(
                pw_req[
                    "shift"
                ]
            ),

        "activation_output_bits":
            4,

        "activation_scale":
            float(
                cfg[
                    "output_scale"
                ]
            ),
    }

    with open(
        OUTPUT_DIR
        / "config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    print()
    print(
        "Generated:"
    )

    for p in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            " -",
            p.name
        )

    print()
    print(
        "=============================================="
    )
    print(
        "READY FOR FPGA INT4 TEST"
    )
    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()