import json
from pathlib import Path
from pprint import pprint

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

SCALES_FILE = (
    FINAL_DIR
    / "deployment_scales.json"
)


with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8"
) as f:
    manifest = json.load(f)


with open(
    SCALES_FILE,
    "r",
    encoding="utf-8"
) as f:
    scales = json.load(f)


print("=" * 80)
print("REMAINING DEPLOYMENT LAYERS")
print("=" * 80)


targets = (
    "block2",
    "block3",
    "block4",
    "head",
    "dense",
    "classifier",
    "logits",
)


for name, cfg in manifest["layers"].items():

    lname = name.lower()

    if any(
        token in lname
        for token in targets
    ):

        print()
        print("-" * 80)
        print(name)
        print("-" * 80)

        pprint(cfg)

        print()

        for key in [
            "kernel_file",
            "bias_file",
            "depthwise_file",
            "pointwise_file",
        ]:

            filename = cfg.get(key)

            if filename is None:
                continue

            path = (
                DEPLOY_DIR
                / filename
            )

            if path.exists():

                arr = np.load(path)

                print(
                    f"{key}:",
                    filename
                )

                print(
                    "  shape =",
                    arr.shape
                )

                print(
                    "  dtype =",
                    arr.dtype
                )

                print(
                    "  range =",
                    int(arr.min()),
                    "...",
                    int(arr.max())
                )


print()
print("=" * 80)
print("RELEVANT TENSOR SCALES")
print("=" * 80)


tensor_scales = scales.get(
    "tensor_scales",
    {}
)


for name, cfg in tensor_scales.items():

    lname = name.lower()

    if any(
        token in lname
        for token in [
            "block2",
            "pool2",
            "block3",
            "pool3",
            "block4",
            "pool4",
            "gap",
            "gmp",
            "concat",
            "dense",
            "head",
            "classifier",
            "logit",
            "output",
        ]
    ):

        print()
        print(name)

        pprint(cfg)


print()
print("=" * 80)
print("SEPARABLE INTERNAL SCALES")
print("=" * 80)


sep_scales = scales.get(
    "separable_internal_scales",
    {}
)


for name, cfg in sep_scales.items():

    if (
        name.startswith("block2")
        or name.startswith("block3")
        or name.startswith("block4")
    ):

        print()
        print(name)

        pprint(cfg)


print()
print("=" * 80)
print("ALL MANIFEST LAYER NAMES")
print("=" * 80)


for i, name in enumerate(
    manifest["layers"].keys()
):

    print(
        f"{i:02d}: {name}"
    )