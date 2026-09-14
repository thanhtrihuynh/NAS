from pathlib import Path

import numpy as np

from deployment_loader import (
    DeploymentLoader
)

from model_block import (
    ResidualInceptionBlock
)

from integer_ops import (
    dense_integer,
    gelu_requant,
    global_average_requant,
    global_max_requant,
    requantize_tensor,
)


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

POOL1_FILE = (
    ROOT
    / "pynq"
    / "real_int4_test"
    / "pool1_int8.npy"
)

OUT_DIR = (
    ROOT
    / "pynq"
    / "full_runtime_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


dep = DeploymentLoader(
    ROOT
)

block = ResidualInceptionBlock(
    dep
)


pool1 = np.load(
    POOL1_FILE
).astype(
    np.int8
)


print("=" * 70)
print("FULL BACKBONE INTEGER TEST")
print("=" * 70)

print(
    "pool1:",
    pool1.shape,
    "range=",
    int(pool1.min()),
    "...",
    int(pool1.max())
)


# =========================================================
# BLOCK 2
# =========================================================

block2 = block.run(
    prefix="block2",

    x=pool1,

    input_scale=
        dep.tensor_scale(
            "pool1"
        ),

    downsample=True
)

pool2 = block2[
    "pooled"
]

print()
print(
    "pool2:",
    pool2.shape,
    "range=",
    int(pool2.min()),
    "...",
    int(pool2.max())
)


if pool2.shape != (
    80,
    12
):
    raise RuntimeError(
        f"pool2 shape sai: "
        f"{pool2.shape}"
    )


# =========================================================
# BLOCK 3
# =========================================================

block3 = block.run(
    prefix="block3",

    x=pool2,

    input_scale=
        dep.tensor_scale(
            "pool2"
        ),

    downsample=True
)

pool3 = block3[
    "pooled"
]

print()
print(
    "pool3:",
    pool3.shape,
    "range=",
    int(pool3.min()),
    "...",
    int(pool3.max())
)


if pool3.shape != (
    40,
    24
):
    raise RuntimeError(
        f"pool3 shape sai: "
        f"{pool3.shape}"
    )


# =========================================================
# BLOCK 4
# =========================================================

block4 = block.run(
    prefix="block4",

    x=pool3,

    input_scale=
        dep.tensor_scale(
            "pool3"
        ),

    downsample=False
)

block4_act = block4[
    "act"
]

print()
print(
    "block4_act:",
    block4_act.shape,
    "range=",
    int(block4_act.min()),
    "...",
    int(block4_act.max())
)


if block4_act.shape != (
    40,
    28
):
    raise RuntimeError(
        f"block4_act shape sai: "
        f"{block4_act.shape}"
    )


# =========================================================
# GLOBAL AVERAGE + GLOBAL MAX
# =========================================================

block4_scale = (
    dep.tensor_scale(
        "block4_act"
    )
)


gap = global_average_requant(
    x=block4_act,

    input_scale=
        block4_scale,

    output_scale=
        dep.tensor_scale(
            "gap"
        ),

    output_bits=
        dep.tensor_bits(
            "gap"
        )
)


gmp = global_max_requant(
    x=block4_act,

    input_scale=
        block4_scale,

    output_scale=
        dep.tensor_scale(
            "gmp"
        ),

    output_bits=
        dep.tensor_bits(
            "gmp"
        )
)


print()
print(
    "gap:",
    gap.shape,
    "range=",
    int(gap.min()),
    "...",
    int(gap.max())
)

print(
    "gmp:",
    gmp.shape,
    "range=",
    int(gmp.min()),
    "...",
    int(gmp.max())
)


if gap.shape != (28,):
    raise RuntimeError(
        "GAP shape sai."
    )

if gmp.shape != (28,):
    raise RuntimeError(
        "GMP shape sai."
    )


# =========================================================
# HEAD CONCAT
# =========================================================

head_scale = (
    dep.tensor_scale(
        "head_concat"
    )
)

head_bits = (
    dep.tensor_bits(
        "head_concat"
    )
)


gap_aligned = requantize_tensor(
    gap,
    dep.tensor_scale(
        "gap"
    ),
    head_scale,
    head_bits
)


gmp_aligned = requantize_tensor(
    gmp,
    dep.tensor_scale(
        "gmp"
    ),
    head_scale,
    head_bits
)


head_concat = np.concatenate(
    [
        gap_aligned,
        gmp_aligned,
    ]
).astype(
    np.int8
)


print()
print(
    "head_concat:",
    head_concat.shape
)


if head_concat.shape != (
    56,
):
    raise RuntimeError(
        "head_concat shape sai."
    )


# =========================================================
# HEAD DENSE 56 -> 24
# =========================================================

head_cfg = dep.layer(
    "head_dense"
)

head_kernel = dep.load_array(
    head_cfg[
        "kernel_file"
    ],
    np.int8
)

head_bias = dep.load_array(
    head_cfg[
        "bias_file"
    ],
    np.int32
)


head_req = head_cfg[
    "acc_to_preact"
]


head_preact = dense_integer(
    x=head_concat,

    kernel=head_kernel,

    bias=head_bias,

    multiplier=int(
        head_req[
            "multiplier"
        ]
    ),

    shift=int(
        head_req[
            "shift"
        ]
    ),

    output_bits=
        dep.tensor_bits(
            "head_bn"
        )
)


head_act = gelu_requant(
    q_in=head_preact,

    input_scale=float(
        head_cfg[
            "preactivation_scale"
        ]
    ),

    output_scale=float(
        head_cfg[
            "output_scale"
        ]
    ),

    output_bits=int(
        head_cfg[
            "output_bits"
        ]
    )
)


print()
print(
    "head_preact:",
    head_preact.shape,
    "range=",
    int(head_preact.min()),
    "...",
    int(head_preact.max())
)

print(
    "head_act:",
    head_act.shape,
    "range=",
    int(head_act.min()),
    "...",
    int(head_act.max())
)


if head_act.shape != (
    24,
):
    raise RuntimeError(
        "head_act shape sai."
    )


# =========================================================
# SAVE
# =========================================================

np.save(
    OUT_DIR
    / "pool2.npy",
    pool2
)

np.save(
    OUT_DIR
    / "pool3.npy",
    pool3
)

np.save(
    OUT_DIR
    / "block4_act.npy",
    block4_act
)

np.save(
    OUT_DIR
    / "gap.npy",
    gap
)

np.save(
    OUT_DIR
    / "gmp.npy",
    gmp
)

np.save(
    OUT_DIR
    / "head_concat.npy",
    head_concat
)

np.save(
    OUT_DIR
    / "head_act.npy",
    head_act
)


print()
print("=" * 70)
print("PASS: POOL1 -> HEAD_ACT COMPLETED")
print("=" * 70)