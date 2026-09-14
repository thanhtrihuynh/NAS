import json
from pathlib import Path

from config import ARTIFACTS_DIR


REQUIRED_SCALES = [
    "__input__",

    "stem_act",

    "block1_concat",
    "block1_out_bn",
    "block1_add",
    "block1_act",
    "pool1",

    "block2_concat",
    "block2_out_bn",
    "block2_add",
    "block2_act",
    "pool2",

    "block3_concat",
    "block3_out_bn",
    "block3_add",
    "block3_act",
    "pool3",

    "block4_concat",
    "block4_out_bn",
    "block4_add",
    "block4_act",

    "head_concat",
    "head_bn",
    "head_act"
]


def main():
    scale_path = (
        ARTIFACTS_DIR
        / "deployment_scales.json"
    )

    if not scale_path.exists():
        raise FileNotFoundError(
            scale_path
        )

    with Path(
        scale_path
    ).open(
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    scales = (
        data[
            "tensor_scales"
        ]
    )

    print()
    print("=" * 70)
    print("FULL INTEGER SCALE CHECK")
    print("=" * 70)

    missing = []

    for name in REQUIRED_SCALES:
        if name in scales:
            info = (
                scales[name]
            )

            print(
                f"OK   {name:<24} "
                f"INT{info['bits']:<2} "
                f"scale={info['scale']:.10f}"
            )

        else:
            print(
                f"MISS {name}"
            )

            missing.append(
                name
            )

    print()
    print("=" * 70)

    if len(missing) == 0:
        print(
            "ALL REQUIRED SCALES FOUND"
        )

    else:
        print(
            "MISSING SCALE COUNT:",
            len(missing)
        )

        print(
            "Missing:"
        )

        for name in missing:
            print(
                " -",
                name
            )


if __name__ == "__main__":
    main()