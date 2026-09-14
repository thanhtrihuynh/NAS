import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

DEPLOY_DIR = (
    ROOT
    / "artifacts"
    / "final_w4a4_p99_9"
    / "deployment_model"
)

MANIFEST_FILE = (
    DEPLOY_DIR
    / "deployment_manifest.json"
)

STEM_TEST_DIR = (
    ROOT
    / "pynq"
    / "real_stem_test"
)

OUTPUT_DIR = (
    ROOT
    / "pynq"
    / "real_sepconv_test"
)


def quantize_symmetric(x, scale, bits):
    if bits == 4:
        qmin = -7
        qmax = 7
    elif bits == 8:
        qmin = -127
        qmax = 127
    else:
        raise ValueError(
            f"Unsupported bits: {bits}"
        )

    q = np.rint(
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


def gelu_exact(x):
    x = np.asarray(
        x,
        dtype=np.float32
    )

    erf_func = np.vectorize(
        math.erf
    )

    y = (
        0.5
        * x
        * (
            1.0
            + erf_func(
                x / math.sqrt(2.0)
            )
        )
    )

    return y.astype(
        np.float32
    )


def round_shift_signed(
    value,
    shift
):
    value = int(value)
    shift = int(shift)

    if shift == 0:
        return value

    bias = (
        1
        << (shift - 1)
    )

    if value >= 0:
        return (
            value + bias
        ) >> shift

    return -(
        (
            (-value) + bias
        )
        >> shift
    )


def saturate(
    value,
    bits
):
    if bits == 4:
        return max(
            -7,
            min(
                7,
                int(value)
            )
        )

    if bits == 8:
        return max(
            -127,
            min(
                127,
                int(value)
            )
        )

    raise ValueError(
        f"Unsupported bits: {bits}"
    )


def depthwise_same_integer(
    x,
    kernel,
    multiplier,
    shift,
    dilation,
    output_bits
):
    length, cin = x.shape

    kernel_size, kernel_cin, depth_multiplier = (
        kernel.shape
    )

    if kernel_cin != cin:
        raise ValueError(
            "Depthwise Cin mismatch: "
            f"input={cin}, kernel={kernel_cin}"
        )

    if depth_multiplier != 1:
        raise ValueError(
            "Chỉ hỗ trợ depth_multiplier = 1."
        )

    pad_left = (
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

    for out_pos in range(
        length
    ):
        for channel in range(
            cin
        ):
            acc = 0

            for k in range(
                kernel_size
            ):
                input_pos = (
                    out_pos
                    + k * dilation
                    - pad_left
                )

                if (
                    input_pos < 0
                    or
                    input_pos >= length
                ):
                    continue

                acc += (
                    int(
                        x[
                            input_pos,
                            channel
                        ]
                    )
                    *
                    int(
                        kernel[
                            k,
                            channel,
                            0
                        ]
                    )
                )

            product = (
                acc
                * int(multiplier)
            )

            q = round_shift_signed(
                product,
                shift
            )

            output[
                out_pos,
                channel
            ] = saturate(
                q,
                output_bits
            )

    return output


def pointwise_integer(
    x,
    kernel,
    bias,
    multiplier,
    shift,
    output_bits
):
    length, cin = x.shape

    if kernel.ndim != 3:
        raise ValueError(
            "Pointwise kernel phải có "
            "shape [1, Cin, Cout]."
        )

    if kernel.shape[0] != 1:
        raise ValueError(
            "Pointwise kernel_size phải = 1."
        )

    if kernel.shape[1] != cin:
        raise ValueError(
            "Pointwise Cin mismatch."
        )

    cout = int(
        kernel.shape[2]
    )

    if bias.shape != (cout,):
        raise ValueError(
            "Bias shape không đúng."
        )

    output = np.zeros(
        (
            length,
            cout
        ),
        dtype=np.int8
    )

    for out_pos in range(
        length
    ):
        for out_ch in range(
            cout
        ):
            acc = int(
                bias[out_ch]
            )

            for in_ch in range(
                cin
            ):
                acc += (
                    int(
                        x[
                            out_pos,
                            in_ch
                        ]
                    )
                    *
                    int(
                        kernel[
                            0,
                            in_ch,
                            out_ch
                        ]
                    )
                )

            product = (
                acc
                * int(multiplier)
            )

            q = round_shift_signed(
                product,
                shift
            )

            output[
                out_pos,
                out_ch
            ] = saturate(
                q,
                output_bits
            )

    return output


def main():
    print(
        "=========================================="
    )
    print(
        "REAL BLOCK1_B1 SEPARABLE CONV GENERATOR"
    )
    print(
        "=========================================="
    )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            MANIFEST_FILE
        )

    stem_expected_file = (
        STEM_TEST_DIR
        / "expected_int8.npy"
    )

    stem_config_file = (
        STEM_TEST_DIR
        / "config.json"
    )

    if not stem_expected_file.exists():
        raise FileNotFoundError(
            "Không tìm thấy kết quả Stem test:\n"
            f"{stem_expected_file}"
        )

    if not stem_config_file.exists():
        raise FileNotFoundError(
            stem_config_file
        )

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    with open(
        stem_config_file,
        "r",
        encoding="utf-8"
    ) as f:
        stem_test_config = json.load(f)

    stem_cfg = (
        manifest["layers"]
        ["stem_conv"]
    )

    sep_cfg = (
        manifest["layers"]
        ["block1_b1_sepconv"]
    )

    print()
    print(
        "Source ECG:"
    )

    print(
        "Label:",
        stem_test_config.get(
            "label"
        )
    )

    print(
        "Record ID:",
        stem_test_config.get(
            "record_id"
        )
    )

    print(
        "Local record:",
        stem_test_config.get(
            "record_local"
        )
    )

    # =====================================================
    # 1. Load Stem preactivation integer output
    # =====================================================

    stem_preact_q = np.load(
        stem_expected_file
    ).astype(
        np.int8
    )

    if stem_preact_q.shape != (
        320,
        8
    ):
        raise ValueError(
            "Stem expected shape phải là "
            f"(320, 8), nhận {stem_preact_q.shape}"
        )

    stem_preact_scale = float(
        stem_cfg[
            "preactivation_scale"
        ]
    )

    stem_act_scale = float(
        stem_cfg[
            "output_scale"
        ]
    )

    print()
    print(
        "Stem preactivation:"
    )

    print(
        "shape:",
        stem_preact_q.shape
    )

    print(
        "scale:",
        stem_preact_scale
    )

    print(
        "range:",
        int(np.min(stem_preact_q)),
        "...",
        int(np.max(stem_preact_q))
    )

    # =====================================================
    # 2. Preactivation INT8 -> float -> GELU -> INT8
    # =====================================================

    stem_preact_float = (
        stem_preact_q.astype(
            np.float32
        )
        * stem_preact_scale
    )

    stem_act_float = gelu_exact(
        stem_preact_float
    )

    input_q = quantize_symmetric(
        stem_act_float,
        stem_act_scale,
        8
    )

    print()
    print(
        "stem_act input for block1_b1:"
    )

    print(
        "shape:",
        input_q.shape
    )

    print(
        "scale:",
        stem_act_scale
    )

    print(
        "range:",
        int(np.min(input_q)),
        "...",
        int(np.max(input_q))
    )

    print(
        "manifest input_scale:",
        float(
            sep_cfg["input_scale"]
        )
    )

    if not np.isclose(
        stem_act_scale,
        float(
            sep_cfg["input_scale"]
        ),
        rtol=0,
        atol=1e-12
    ):
        raise ValueError(
            "stem_act output scale không khớp "
            "block1_b1 input scale."
        )

    # =====================================================
    # 3. Load real deployment weights
    # =====================================================

    depthwise = np.load(
        DEPLOY_DIR
        / sep_cfg[
            "depthwise_file"
        ]
    ).astype(
        np.int8
    )

    pointwise = np.load(
        DEPLOY_DIR
        / sep_cfg[
            "pointwise_file"
        ]
    ).astype(
        np.int8
    )

    bias = np.load(
        DEPLOY_DIR
        / sep_cfg[
            "bias_file"
        ]
    ).astype(
        np.int32
    )

    print()
    print(
        "Depthwise shape:",
        depthwise.shape
    )

    print(
        "Depthwise range:",
        int(np.min(depthwise)),
        "...",
        int(np.max(depthwise))
    )

    print()
    print(
        "Pointwise shape:",
        pointwise.shape
    )

    print(
        "Pointwise range:",
        int(np.min(pointwise)),
        "...",
        int(np.max(pointwise))
    )

    print()
    print(
        "Bias shape:",
        bias.shape
    )

    expected_dw_shape = tuple(
        sep_cfg[
            "depthwise_shape"
        ]
    )

    expected_pw_shape = tuple(
        sep_cfg[
            "pointwise_shape"
        ]
    )

    if depthwise.shape != expected_dw_shape:
        raise ValueError(
            f"Depthwise shape mismatch: "
            f"{depthwise.shape} != "
            f"{expected_dw_shape}"
        )

    if pointwise.shape != expected_pw_shape:
        raise ValueError(
            f"Pointwise shape mismatch: "
            f"{pointwise.shape} != "
            f"{expected_pw_shape}"
        )

    # =====================================================
    # 4. Depthwise integer golden
    # =====================================================

    dw_req = (
        sep_cfg[
            "depthwise_requant"
        ]
    )

    dw_multiplier = int(
        dw_req[
            "multiplier"
        ]
    )

    dw_shift = int(
        dw_req[
            "shift"
        ]
    )

    dilation = int(
        sep_cfg[
            "dilation"
        ]
    )

    expected_dw = (
        depthwise_same_integer(
            x=input_q,

            kernel=depthwise,

            multiplier=
                dw_multiplier,

            shift=
                dw_shift,

            dilation=
                dilation,

            output_bits=8
        )
    )

    print()
    print(
        "Depthwise requant:"
    )

    print(
        "multiplier:",
        dw_multiplier
    )

    print(
        "shift:",
        dw_shift
    )

    print(
        "dilation:",
        dilation
    )

    print(
        "expected shape:",
        expected_dw.shape
    )

    print(
        "expected range:",
        int(np.min(expected_dw)),
        "...",
        int(np.max(expected_dw))
    )

    # =====================================================
    # 5. Pointwise integer golden
    # =====================================================

    pw_req = (
        sep_cfg[
            "acc_to_preact"
        ]
    )

    pw_multiplier = int(
        pw_req[
            "multiplier"
        ]
    )

    pw_shift = int(
        pw_req[
            "shift"
        ]
    )

    expected_pw = (
        pointwise_integer(
            x=expected_dw,

            kernel=pointwise,

            bias=bias,

            multiplier=
                pw_multiplier,

            shift=
                pw_shift,

            output_bits=8
        )
    )

    print()
    print(
        "Pointwise requant:"
    )

    print(
        "multiplier:",
        pw_multiplier
    )

    print(
        "shift:",
        pw_shift
    )

    print(
        "expected shape:",
        expected_pw.shape
    )

    print(
        "expected range:",
        int(np.min(expected_pw)),
        "...",
        int(np.max(expected_pw))
    )

    # =====================================================
    # 6. Save package
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR
        / "stem_preact_int8.npy",
        stem_preact_q
    )

    np.save(
        OUTPUT_DIR
        / "stem_act_float.npy",
        stem_act_float
    )

    np.save(
        OUTPUT_DIR
        / "input_int8.npy",
        input_q
    )

    np.save(
        OUTPUT_DIR
        / "depthwise_int8.npy",
        depthwise
    )

    np.save(
        OUTPUT_DIR
        / "pointwise_int8.npy",
        pointwise
    )

    np.save(
        OUTPUT_DIR
        / "bias_int32.npy",
        bias
    )

    np.save(
        OUTPUT_DIR
        / "expected_depthwise.npy",
        expected_dw
    )

    np.save(
        OUTPUT_DIR
        / "expected_pointwise.npy",
        expected_pw
    )

    config = {
        "source":
            "real_stem_test",

        "label":
            stem_test_config.get(
                "label"
            ),

        "record_id":
            stem_test_config.get(
                "record_id"
            ),

        "layer":
            "block1_b1_sepconv",

        "bits":
            8,

        "input_len":
            320,

        "cin":
            8,

        "cout":
            4,

        "input_scale":
            float(
                sep_cfg[
                    "input_scale"
                ]
            ),

        "kernel_size":
            int(
                depthwise.shape[0]
            ),

        "dilation":
            dilation,

        "same":
            True,

        "depthwise_multiplier":
            dw_multiplier,

        "depthwise_shift":
            dw_shift,

        "depthwise_output_scale":
            float(
                sep_cfg[
                    "depthwise_output_scale"
                ]
            ),

        "pointwise_multiplier":
            pw_multiplier,

        "pointwise_shift":
            pw_shift,

        "preactivation_scale":
            float(
                sep_cfg[
                    "preactivation_scale"
                ]
            ),

        "output_bits":
            8
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
        "Generated package:"
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
        "=========================================="
    )
    print(
        "REAL SEPCONV TEST PACKAGE READY"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()