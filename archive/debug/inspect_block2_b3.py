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

POOL1_FILE = (
    ROOT
    / "pynq"
    / "real_int4_test"
    / "pool1_int8.npy"
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


layer_name = "block2_b3_sepconv"

cfg = manifest[
    "layers"
][layer_name]


print("=" * 70)
print("BLOCK2_B3 SEPARABLE CONV")
print("=" * 70)

pprint(cfg)


print()
print("=" * 70)
print("RELEVANT TENSOR SCALES")
print("=" * 70)


tensor_names = [
    "pool1",
    "block2_b3_sepconv",
    "block2_b3_bn",
    "block2_b3_act",
    "block2_concat",
]


for name in tensor_names:

    item = (
        scales
        .get(
            "tensor_scales",
            {}
        )
        .get(name)
    )

    print()
    print(name)

    if item is None:
        print("NOT FOUND")
    else:
        pprint(item)


print()
print("=" * 70)
print("SEPARABLE INTERNAL SCALE")
print("=" * 70)


internal = (
    scales
    .get(
        "separable_internal_scales",
        {}
    )
    .get(layer_name)
)

if internal is None:
    print(
        "block2_b3_sepconv internal scale NOT FOUND"
    )
else:
    pprint(internal)


print()
print("=" * 70)
print("POOL1 REAL TEST")
print("=" * 70)


pool1 = np.load(
    POOL1_FILE
).astype(
    np.int8
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


pool1_cfg = (
    scales[
        "tensor_scales"
    ][
        "pool1"
    ]
)


print(
    "bits:",
    pool1_cfg[
        "bits"
    ]
)

print(
    "scale:",
    pool1_cfg[
        "scale"
    ]
)


print()
print("=" * 70)
print("WEIGHT FILES")
print("=" * 70)


depthwise = np.load(
    DEPLOY_DIR
    / cfg[
        "depthwise_file"
    ]
)

pointwise = np.load(
    DEPLOY_DIR
    / cfg[
        "pointwise_file"
    ]
)

bias = np.load(
    DEPLOY_DIR
    / cfg[
        "bias_file"
    ]
)


print(
    "Depthwise:",
    depthwise.shape,
    depthwise.dtype,
    "range=",
    int(depthwise.min()),
    "...",
    int(depthwise.max())
)

print(
    "Pointwise:",
    pointwise.shape,
    pointwise.dtype,
    "range=",
    int(pointwise.min()),
    "...",
    int(pointwise.max())
)

print(
    "Bias:",
    bias.shape,
    bias.dtype,
    "range=",
    int(bias.min()),
    "...",
    int(bias.max())
)


print()
print("=" * 70)
print("IMPORTANT PARAMETERS")
print("=" * 70)

print(
    "layer bits:",
    cfg.get(
        "bits"
    )
)

print(
    "input_source:",
    cfg.get(
        "input_source"
    )
)

print(
    "input_scale:",
    cfg.get(
        "input_scale"
    )
)

print(
    "depthwise_shape:",
    cfg.get(
        "depthwise_shape"
    )
)

print(
    "depthwise_output_scale:",
    cfg.get(
        "depthwise_output_scale"
    )
)

print(
    "depthwise_requant:"
)

pprint(
    cfg.get(
        "depthwise_requant"
    )
)

print(
    "pointwise_shape:",
    cfg.get(
        "pointwise_shape"
    )
)

print(
    "preactivation_scale:",
    cfg.get(
        "preactivation_scale"
    )
)

print(
    "acc_to_preact:"
)

pprint(
    cfg.get(
        "acc_to_preact"
    )
)

print(
    "output_bits:",
    cfg.get(
        "output_bits"
    )
)

print(
    "output_scale:",
    cfg.get(
        "output_scale"
    )
)

print(
    "output_mode:",
    cfg.get(
        "output_mode"
    )
)