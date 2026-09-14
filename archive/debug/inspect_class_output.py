import json
from pathlib import Path
from pprint import pprint

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


with open(
    MANIFEST_FILE,
    "r",
    encoding="utf-8"
) as f:
    manifest = json.load(f)


cfg = manifest[
    "layers"
][
    "class_output"
]


print("=" * 70)
print("CLASS OUTPUT")
print("=" * 70)

pprint(cfg)


print()
print("=" * 70)
print("WEIGHTS")
print("=" * 70)


kernel = np.load(
    DEPLOY_DIR
    / cfg[
        "kernel_file"
    ]
)

bias = np.load(
    DEPLOY_DIR
    / cfg[
        "bias_file"
    ]
)


print(
    "Kernel shape:",
    kernel.shape
)

print(
    "Kernel dtype:",
    kernel.dtype
)

print(
    "Kernel range:",
    int(kernel.min()),
    "...",
    int(kernel.max())
)


print()

print(
    "Bias shape:",
    bias.shape
)

print(
    "Bias dtype:",
    bias.dtype
)

print(
    "Bias range:",
    int(bias.min()),
    "...",
    int(bias.max())
)