import json
from pathlib import Path

import numpy as np

from runtime_to_pool1 import RuntimeToPool1


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

STEM_DIR = (
    ROOT
    / "pynq"
    / "real_stem_test"
)

OUT_DIR = (
    ROOT
    / "pynq"
    / "real_int4_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


STEM_FILE = (
    STEM_DIR
    / "expected_int8.npy"
)

if not STEM_FILE.exists():
    raise FileNotFoundError(
        f"Không tìm thấy: {STEM_FILE}"
    )


stem_preact = np.load(
    STEM_FILE
).astype(
    np.int8
)


print("=" * 70)
print("BLOCK1 -> POOL1 INTEGER RUNTIME")
print("=" * 70)

print(
    "Stem preactivation:",
    stem_preact.shape,
    stem_preact.dtype,
    "range=",
    int(stem_preact.min()),
    "...",
    int(stem_preact.max())
)


runtime = RuntimeToPool1(
    ROOT
)

result = runtime.run(
    stem_preact
)


print()
print("=" * 70)
print("TRACE SUMMARY")
print("=" * 70)


trace_names = [
    "stem_act",
    "block1_b1_depthwise",
    "block1_b1_preact",
    "block1_b1_act",
    "block1_b2_depthwise",
    "block1_b2_preact",
    "block1_b2_act",
    "block1_b3_depthwise",
    "block1_b3_preact",
    "block1_b3_act",
    "block1_pool",
    "block1_pool_preact",
    "block1_pool_act",
    "block1_concat",
    "block1_out_bn",
    "block1_add",
    "block1_act",
    "pool1",
]


for name in trace_names:
    x = result[name]

    print(
        f"{name:24s}",
        f"shape={str(x.shape):12s}",
        f"dtype={str(x.dtype):6s}",
        f"range={int(x.min()):4d}"
        f" ... {int(x.max()):4d}"
    )


pool1 = result[
    "pool1"
]

pool1_scale = float(
    result[
        "pool1_scale"
    ]
)


if pool1.shape != (
    160,
    8
):
    raise RuntimeError(
        f"POOL1 shape sai: "
        f"{pool1.shape}"
    )


if pool1.dtype != np.int8:
    raise RuntimeError(
        f"POOL1 dtype sai: "
        f"{pool1.dtype}"
    )


expected_pool1_scale = (
    0.1295149495282511
)

if not np.isclose(
    pool1_scale,
    expected_pool1_scale,
    rtol=0.0,
    atol=1e-12
):
    raise RuntimeError(
        "POOL1 scale sai: "
        f"{pool1_scale}"
    )


pool1_file = (
    OUT_DIR
    / "pool1_int8.npy"
)

np.save(
    pool1_file,
    pool1
)


trace_file = (
    OUT_DIR
    / "block1_trace.npz"
)

np.savez(
    trace_file,
    **{
        name: result[name]
        for name in trace_names
    }
)


source_config = {}

stem_config_file = (
    STEM_DIR
    / "config.json"
)

if stem_config_file.exists():

    with open(
        stem_config_file,
        "r",
        encoding="utf-8"
    ) as f:
        source_config = json.load(f)


config = {
    "source":
        "real_stem_test",

    "source_config":
        source_config,

    "pool1_shape":
        list(pool1.shape),

    "pool1_dtype":
        str(pool1.dtype),

    "pool1_scale":
        pool1_scale,

    "pool1_bits":
        8,

    "pool1_min":
        int(pool1.min()),

    "pool1_max":
        int(pool1.max()),

    "next_target_layer":
        "block2_b3_sepconv",
}


config_file = (
    OUT_DIR
    / "config.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=4,
        ensure_ascii=False
    )


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(
    "pool1 shape :",
    pool1.shape
)

print(
    "pool1 dtype :",
    pool1.dtype
)

print(
    "pool1 range :",
    int(pool1.min()),
    "...",
    int(pool1.max())
)

print(
    "pool1 scale :",
    pool1_scale
)

print()
print(
    "Saved:",
    pool1_file
)

print(
    "Saved:",
    trace_file
)

print(
    "Saved:",
    config_file
)

print()
print(
    "PASS: BLOCK1 -> POOL1 COMPLETED"
)