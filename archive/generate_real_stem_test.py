import argparse
import json
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

OUTPUT_DIR = (
    ROOT
    / "pynq"
    / "real_stem_test"
)


def round_shift_signed(value, shift):
    value = int(value)
    shift = int(shift)

    if shift == 0:
        return value

    bias = 1 << (shift - 1)

    if value >= 0:
        return (value + bias) >> shift

    return -(((-value) + bias) >> shift)


def saturate_int8(value):
    return max(
        -127,
        min(
            127,
            int(value)
        )
    )


def quantize_symmetric_int8(x, scale):
    q = np.rint(
        x / scale
    )

    q = np.clip(
        q,
        -127,
        127
    )

    return q.astype(
        np.int8
    )


def ensure_single_ecg(arr, index):
    arr = np.asarray(arr)

    print(
        "Loaded ECG array shape:",
        arr.shape
    )

    if arr.ndim == 1:
        x = arr

    elif arr.ndim == 2:
        # Một sample [320, 1]
        if arr.shape == (320, 1):
            x = arr

        # Dataset [N, 320]
        elif arr.shape[1] == 320:
            x = arr[index]

        else:
            raise ValueError(
                f"Không hiểu shape ECG: {arr.shape}"
            )

    elif arr.ndim == 3:
        # Dataset [N, 320, 1]
        if (
            arr.shape[1] == 320
            and arr.shape[2] == 1
        ):
            x = arr[index]

        else:
            raise ValueError(
                f"Không hiểu shape ECG: {arr.shape}"
            )

    else:
        raise ValueError(
            f"Không hỗ trợ ndim={arr.ndim}"
        )

    x = np.asarray(
        x,
        dtype=np.float32
    )

    if x.ndim == 1:
        x = x[:, None]

    if x.shape != (320, 1):
        raise ValueError(
            f"ECG cuối phải có shape (320,1), "
            f"nhưng nhận {x.shape}"
        )

    return x


def zscore_segment(x):
    mean = np.mean(
        x,
        dtype=np.float64
    )

    std = np.std(
        x,
        dtype=np.float64
    )

    if std < 1e-12:
        raise ValueError(
            "ECG có standard deviation gần 0."
        )

    return (
        (x - mean) / std
    ).astype(
        np.float32
    )


def conv1d_same_integer(
    x,
    kernel,
    bias,
    multiplier,
    shift
):
    input_len, cin = x.shape

    k_size, kernel_cin, cout = (
        kernel.shape
    )

    if kernel_cin != cin:
        raise ValueError(
            "Cin input và kernel không khớp."
        )

    pad_left = (
        k_size - 1
    ) // 2

    output = np.zeros(
        (input_len, cout),
        dtype=np.int8
    )

    for out_pos in range(input_len):

        for out_ch in range(cout):

            acc = int(
                bias[out_ch]
            )

            for k in range(k_size):

                input_pos = (
                    out_pos
                    + k
                    - pad_left
                )

                if (
                    input_pos < 0
                    or input_pos >= input_len
                ):
                    continue

                for in_ch in range(cin):

                    acc += (
                        int(
                            x[
                                input_pos,
                                in_ch
                            ]
                        )
                        *
                        int(
                            kernel[
                                k,
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
            ] = saturate_int8(
                q
            )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ecg",
        required=True,
        help=(
            "File .npy chứa ECG hoặc "
            "dataset ECG."
        )
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index sample nếu file chứa N samples."
    )

    parser.add_argument(
        "--zscore",
        action="store_true",
        help=(
            "Dùng nếu ECG chưa được z-score. "
            "Không dùng nếu X đã là dữ liệu "
            "preprocessed của project."
        )
    )

    args = parser.parse_args()

    print(
        "========================================"
    )
    print(
        "GENERATE REAL STEM FPGA TEST"
    )
    print(
        "========================================"
    )

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    stem = (
        manifest["layers"]
        ["stem_conv"]
    )

    input_scale = float(
        stem["input_scale"]
    )

    weight_scale = float(
        stem["weight_scale"]
    )

    preactivation_scale = float(
        stem["preactivation_scale"]
    )

    kernel_file = (
        DEPLOY_DIR
        / stem["kernel_file"]
    )

    bias_file = (
        DEPLOY_DIR
        / stem["bias_file"]
    )

    kernel = np.load(
        kernel_file
    ).astype(
        np.int8
    )

    bias = np.load(
        bias_file
    ).astype(
        np.int32
    )

    print()
    print(
        "Kernel:",
        kernel_file
    )

    print(
        "Kernel shape:",
        kernel.shape
    )

    print(
        "Bias shape:",
        bias.shape
    )

    if tuple(
        kernel.shape
    ) != (7, 1, 8):
        raise ValueError(
            f"Kernel shape không đúng: "
            f"{kernel.shape}"
        )

    real_multiplier = (
        input_scale
        * weight_scale
        / preactivation_scale
    )

    shift = 31

    multiplier = int(
        round(
            real_multiplier
            * (
                1 << shift
            )
        )
    )

    print()
    print(
        "Input scale:",
        input_scale
    )

    print(
        "Weight scale:",
        weight_scale
    )

    print(
        "Preactivation scale:",
        preactivation_scale
    )

    print(
        "Real multiplier:",
        real_multiplier
    )

    print(
        "Integer multiplier:",
        multiplier
    )

    print(
        "Shift:",
        shift
    )

    if multiplier != 12041293:
        print()
        print(
            "WARNING: multiplier khác "
            "golden value 12041293"
        )

    ecg_array = np.load(
        args.ecg
    )

    x_float = ensure_single_ecg(
        ecg_array,
        args.index
    )

    if args.zscore:

        print()
        print(
            "Applying per-segment z-score..."
        )

        x_float = zscore_segment(
            x_float
        )

    print()
    print(
        "Selected ECG:"
    )

    print(
        "shape =",
        x_float.shape
    )

    print(
        "min   =",
        float(
            np.min(x_float)
        )
    )

    print(
        "max   =",
        float(
            np.max(x_float)
        )
    )

    print(
        "mean  =",
        float(
            np.mean(x_float)
        )
    )

    print(
        "std   =",
        float(
            np.std(x_float)
        )
    )

    x_int8 = quantize_symmetric_int8(
        x_float,
        input_scale
    )

    print()
    print(
        "Quantized input:"
    )

    print(
        "shape =",
        x_int8.shape
    )

    print(
        "min   =",
        int(
            np.min(x_int8)
        )
    )

    print(
        "max   =",
        int(
            np.max(x_int8)
        )
    )

    expected = (
        conv1d_same_integer(
            x=x_int8,
            kernel=kernel,
            bias=bias,
            multiplier=multiplier,
            shift=shift
        )
    )

    print()
    print(
        "Expected stem output:"
    )

    print(
        "shape =",
        expected.shape
    )

    print(
        "min   =",
        int(
            np.min(expected)
        )
    )

    print(
        "max   =",
        int(
            np.max(expected)
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_DIR
        / "input_float.npy",
        x_float
    )

    np.save(
        OUTPUT_DIR
        / "input_int8.npy",
        x_int8
    )

    np.save(
        OUTPUT_DIR
        / "kernel_int8.npy",
        kernel
    )

    np.save(
        OUTPUT_DIR
        / "bias_int32.npy",
        bias
    )

    np.save(
        OUTPUT_DIR
        / "expected_int8.npy",
        expected
    )

    config = {
        "layer": "stem_conv",
        "input_len": 320,
        "cin": 1,
        "cout": 8,

        "kernel_size": 7,
        "stride": 1,
        "dilation": 1,
        "same": True,

        "input_bits": 8,
        "weight_bits": 8,
        "output_bits": 8,

        "input_scale": input_scale,
        "weight_scale": weight_scale,

        "preactivation_scale":
            preactivation_scale,

        "real_multiplier":
            real_multiplier,

        "multiplier":
            multiplier,

        "shift":
            shift,

        "sample_index":
            args.index,

        "zscore_applied":
            bool(args.zscore)
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

    np.savetxt(
        OUTPUT_DIR
        / "input_int8.txt",
        x_int8.reshape(-1),
        fmt="%d"
    )

    np.savetxt(
        OUTPUT_DIR
        / "expected_int8.txt",
        expected.reshape(-1),
        fmt="%d"
    )

    print()
    print(
        "Generated:"
    )

    for p in OUTPUT_DIR.iterdir():
        print(
            " -",
            p.name
        )

    print()
    print(
        "========================================"
    )
    print(
        "REAL STEM TEST PACKAGE READY"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()