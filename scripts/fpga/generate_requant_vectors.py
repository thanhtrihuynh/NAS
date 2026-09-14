import argparse
import json
from pathlib import Path

import numpy as np

from config import (
    ARTIFACTS_DIR,
    PROJECT_ROOT
)

from quantization.fixed_point import (
    rounding_right_shift,
    qrange
)


DEFAULT_DEPLOYMENT_DIR = (
    ARTIFACTS_DIR
    / "final_w4a4_p99_9"
    / "deployment_model"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "rtl_vectors"
    / "requant"
)

NUM_RANDOM_VALUES = 256
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deployment-dir",
        type=str,
        default=str(
            DEFAULT_DEPLOYMENT_DIR
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        )
    )

    return parser.parse_args()


def requant_reference(
    accumulator,
    multiplier,
    shift,
    bits
):
    accumulator = np.asarray(
        accumulator,
        dtype=np.int64
    )

    product = (
        accumulator
        * int(multiplier)
    )

    shifted = rounding_right_shift(
        product,
        int(shift)
    )

    qmin, qmax = qrange(
        bits
    )

    output = np.clip(
        shifted,
        qmin,
        qmax
    )

    return output.astype(
        np.int64
    )


def collect_requant_configs(
    manifest
):
    configs = []

    for layer_name, info in (
        manifest["layers"].items()
    ):
        layer_bits = int(
            info["bits"]
        )

        output_bits = int(
            info.get(
                "output_bits",
                layer_bits
            )
        )

        if (
            "depthwise_requant"
            in info
        ):
            rq = info[
                "depthwise_requant"
            ]

            configs.append({
                "name":
                    f"{layer_name}_depthwise",

                "source_layer":
                    layer_name,

                "stage":
                    "depthwise",

                "multiplier":
                    int(
                        rq[
                            "multiplier"
                        ]
                    ),

                "shift":
                    int(
                        rq[
                            "shift"
                        ]
                    ),

                "bits":
                    layer_bits
            })

        if (
            "acc_to_preact"
            in info
        ):
            rq = info[
                "acc_to_preact"
            ]

            configs.append({
                "name":
                    f"{layer_name}_preact",

                "source_layer":
                    layer_name,

                "stage":
                    "preactivation",

                "multiplier":
                    int(
                        rq[
                            "multiplier"
                        ]
                    ),

                "shift":
                    int(
                        rq[
                            "shift"
                        ]
                    ),

                "bits":
                    output_bits
            })

        if (
            "acc_to_output"
            in info
        ):
            rq = info[
                "acc_to_output"
            ]

            configs.append({
                "name":
                    f"{layer_name}_output",

                "source_layer":
                    layer_name,

                "stage":
                    "output",

                "multiplier":
                    int(
                        rq[
                            "multiplier"
                        ]
                    ),

                "shift":
                    int(
                        rq[
                            "shift"
                        ]
                    ),

                "bits":
                    output_bits
            })

    return configs


def make_test_values(
    rng
):
    corner_values = np.array(
        [
            -(2 ** 31),
            -(2 ** 31) + 1,

            -100000000,
            -10000000,
            -1000000,
            -100000,
            -10000,
            -1000,
            -100,
            -10,
            -3,
            -2,
            -1,

            0,

            1,
            2,
            3,
            10,
            100,
            1000,
            10000,
            100000,
            1000000,
            10000000,
            100000000,

            (2 ** 31) - 2,
            (2 ** 31) - 1
        ],
        dtype=np.int64
    )

    small_values = np.arange(
        -32,
        33,
        dtype=np.int64
    )

    random_values = rng.integers(
        low=-(2 ** 30),
        high=(2 ** 30),
        size=NUM_RANDOM_VALUES,
        dtype=np.int64
    )

    return np.unique(
        np.concatenate(
            [
                corner_values,
                small_values,
                random_values
            ]
        )
    )


def save_decimal_file(
    path,
    values
):
    with Path(path).open(
        "w",
        encoding="utf-8"
    ) as f:
        for value in values:
            f.write(
                f"{int(value)}\n"
            )


def save_hex_file(
    path,
    values,
    width_bits
):
    mask = (
        (1 << width_bits)
        - 1
    )

    width_hex = (
        width_bits + 3
    ) // 4

    with Path(path).open(
        "w",
        encoding="utf-8"
    ) as f:
        for value in values:
            encoded = (
                int(value)
                & mask
            )

            f.write(
                f"{encoded:0{width_hex}X}\n"
            )


def main():
    args = parse_args()

    deployment_dir = Path(
        args.deployment_dir
    )

    output_dir = Path(
        args.output_dir
    )

    manifest_path = (
        deployment_dir
        / "deployment_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy manifest:\n"
            f"{manifest_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    with manifest_path.open(
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    configs = collect_requant_configs(
        manifest
    )

    rng = np.random.default_rng(
        SEED
    )

    accumulator_values = (
        make_test_values(
            rng
        )
    )

    summary = []

    print()
    print("=" * 90)
    print("GENERATING REQUANTIZATION GOLDEN VECTORS")
    print("=" * 90)

    print(
        "Deployment:"
    )

    print(
        deployment_dir
    )

    print()

    for index, config in enumerate(
        configs
    ):
        name = config[
            "name"
        ]

        multiplier = config[
            "multiplier"
        ]

        shift = config[
            "shift"
        ]

        bits = config[
            "bits"
        ]

        expected = requant_reference(
            accumulator_values,
            multiplier,
            shift,
            bits
        )

        case_dir = (
            output_dir
            / (
                f"{index:02d}_"
                f"{name}"
            )
        )

        case_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            case_dir
            / "accumulator.npy",
            accumulator_values
        )

        np.save(
            case_dir
            / "expected.npy",
            expected
        )

        save_decimal_file(
            case_dir
            / "accumulator.txt",
            accumulator_values
        )

        save_decimal_file(
            case_dir
            / "expected.txt",
            expected
        )

        save_hex_file(
            case_dir
            / "accumulator.hex",
            accumulator_values,
            32
        )

        save_hex_file(
            case_dir
            / "expected.hex",
            expected,
            bits
        )

        case_config = {
            "index":
                index,

            "name":
                name,

            "source_layer":
                config[
                    "source_layer"
                ],

            "stage":
                config[
                    "stage"
                ],

            "multiplier":
                multiplier,

            "shift":
                shift,

            "output_bits":
                bits,

            "num_vectors":
                int(
                    len(
                        accumulator_values
                    )
                )
        }

        with (
            case_dir
            / "config.json"
        ).open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                case_config,
                f,
                indent=2,
                ensure_ascii=False
            )

        summary.append(
            case_config
        )

        print(
            f"{index:02d}  "
            f"{name:<42} "
            f"M={multiplier:<12} "
            f"shift={shift:<3} "
            f"INT{bits}"
        )

    summary_data = {
        "source_deployment":
            str(
                deployment_dir
            ),

        "seed":
            SEED,

        "vectors_per_case":
            int(
                len(
                    accumulator_values
                )
            ),

        "cases":
            summary
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 90)

    print(
        "Requant cases:",
        len(configs)
    )

    print(
        "Vectors/case:",
        len(
            accumulator_values
        )
    )

    print(
        "Output:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()