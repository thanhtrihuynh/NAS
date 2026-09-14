import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

DATA_DIR = ROOT / "datasets"

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

SEGMENT_LEN = 320
LABELS = ["N", "L", "R", "V", "A"]


def get_record_id(record_path):
    s = str(record_path).replace("\\", "/")
    return Path(s).stem


def resolve_local_record(record_id):
    candidates = [
        DATA_DIR / record_id,
        DATA_DIR / "x_mitdb" / record_id,
    ]

    for base in candidates:
        hea = base.with_suffix(".hea")
        dat = base.with_suffix(".dat")

        if hea.exists() and dat.exists():
            return base

    return None


def fix_length(x):
    x = np.asarray(
        x,
        dtype=np.float32
    ).reshape(-1)

    if len(x) == SEGMENT_LEN:
        return x

    if len(x) > SEGMENT_LEN:
        center = len(x) // 2
        start = center - SEGMENT_LEN // 2

        return x[
            start:
            start + SEGMENT_LEN
        ]

    pad_left = (
        SEGMENT_LEN - len(x)
    ) // 2

    pad_right = (
        SEGMENT_LEN
        - len(x)
        - pad_left
    )

    return np.pad(
        x,
        (pad_left, pad_right),
        mode="constant"
    )


def normalize_segment(x):
    x = np.asarray(
        x,
        dtype=np.float32
    )

    mean = np.mean(x)
    std = np.std(x)

    if std < 1e-6:
        std = 1.0

    return (
        (x - mean) / std
    ).astype(np.float32)


def quantize_int8(x, scale):
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


def round_shift_signed(value, shift):
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
        )
        >> shift
    )


def saturate_int8(value):
    return max(
        -127,
        min(
            127,
            int(value)
        )
    )


def conv1d_same_integer(
    x,
    kernel,
    bias,
    multiplier,
    shift
):
    input_len, cin = x.shape

    kernel_size, kernel_cin, cout = (
        kernel.shape
    )

    if kernel_cin != cin:
        raise ValueError(
            "Cin input và kernel không khớp."
        )

    pad_left = (
        kernel_size - 1
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

            for k in range(kernel_size):

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


def select_valid_sample(
    df,
    label,
    index
):
    rows = df[
        df["labels"] == label
    ].reset_index(
        drop=True
    )

    valid_rows = []

    for _, row in rows.iterrows():

        record_id = get_record_id(
            row["record_path"]
        )

        local_base = resolve_local_record(
            record_id
        )

        if local_base is not None:

            valid_rows.append(
                (
                    row,
                    record_id,
                    local_base
                )
            )

    if not valid_rows:
        raise RuntimeError(
            f"Không tìm thấy sample local "
            f"cho label {label}"
        )

    if index >= len(valid_rows):
        raise IndexError(
            f"index={index}, nhưng chỉ có "
            f"{len(valid_rows)} sample hợp lệ."
        )

    return valid_rows[index]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        choices=LABELS,
        default="N"
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0
    )

    args = parser.parse_args()

    print(
        "======================================"
    )
    print(
        "REAL MIT-BIH STEM FPGA TEST GENERATOR"
    )
    print(
        "======================================"
    )

    val_csv = (
        DATA_DIR
        / "val.csv"
    )

    print()
    print(
        "Validation CSV:"
    )
    print(
        val_csv
    )

    df = pd.read_csv(
        val_csv
    )

    row, record_id, local_base = (
        select_valid_sample(
            df,
            args.label,
            args.index
        )
    )

    print()
    print(
        "Selected LOCAL validation sample:"
    )

    print(
        "Label:",
        row["labels"]
    )

    print(
        "Record ID:",
        record_id
    )

    print(
        "Local record:",
        local_base
    )

    print(
        "Start:",
        int(row["start"])
    )

    print(
        "End:",
        int(row["end"])
    )

    signal, fields = wfdb.rdsamp(
        str(local_base)
    )

    signal = signal.astype(
        np.float32
    )

    sig_names = fields.get(
        "sig_name",
        []
    )

    channel_name = str(
        row.get(
            "channel",
            "MLII"
        )
    )

    if channel_name in sig_names:

        channel_idx = (
            sig_names.index(
                channel_name
            )
        )

    else:

        channel_idx = 0

        print()
        print(
            f"WARNING: channel "
            f"{channel_name} không tồn tại."
        )

        if sig_names:
            print(
                "Using:",
                sig_names[channel_idx]
            )

    start = int(
        row["start"]
    )

    end = int(
        row["end"]
    )

    raw_segment = signal[
        start:end + 1,
        channel_idx
    ]

    print()
    print(
        "Raw segment length:",
        len(raw_segment)
    )

    x_float = fix_length(
        raw_segment
    )

    x_float = normalize_segment(
        x_float
    )

    x_float = x_float[
        :,
        None
    ]

    print()
    print(
        "Preprocessed shape:",
        x_float.shape
    )

    print(
        "Mean:",
        float(
            np.mean(x_float)
        )
    )

    print(
        "Std:",
        float(
            np.std(x_float)
        )
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

    kernel = np.load(
        DEPLOY_DIR
        / stem["kernel_file"]
    ).astype(
        np.int8
    )

    bias = np.load(
        DEPLOY_DIR
        / stem["bias_file"]
    ).astype(
        np.int32
    )

    print()
    print(
        "Kernel shape:",
        kernel.shape
    )

    print(
        "Bias shape:",
        bias.shape
    )

    x_int8 = quantize_int8(
        x_float,
        input_scale
    )

    print()
    print(
        "Input INT8 range:",
        int(np.min(x_int8)),
        "...",
        int(np.max(x_int8))
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
        "Multiplier:",
        multiplier
    )

    print(
        "Shift:",
        shift
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
        "Expected output shape:",
        expected.shape
    )

    print(
        "Expected range:",
        int(np.min(expected)),
        "...",
        int(np.max(expected))
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
        "source":
            "MIT-BIH local validation",

        "label":
            str(row["labels"]),

        "record_id":
            record_id,

        "record_local":
            str(local_base),

        "channel":
            channel_name,

        "start":
            start,

        "end":
            end,

        "layer":
            "stem_conv",

        "input_len":
            320,

        "cin":
            1,

        "cout":
            8,

        "kernel_size":
            7,

        "dilation":
            1,

        "same":
            True,

        "input_scale":
            input_scale,

        "weight_scale":
            weight_scale,

        "preactivation_scale":
            preactivation_scale,

        "multiplier":
            multiplier,

        "shift":
            shift,

        "input_bits":
            8,

        "weight_bits":
            8,

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
        "READY FOR PYNQ"
    )


if __name__ == "__main__":
    main()