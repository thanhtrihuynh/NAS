import json
from pathlib import Path

import numpy as np

from deployment_loader import (
    DeploymentLoader
)

from runtime_to_pool1 import (
    RuntimeToPool1
)

from model_block import (
    ResidualInceptionBlock
)

from integer_ops import (
    conv1d_same,
    dense_integer,
    dense_logits_int32,
    gelu_requant,
    global_average_requant,
    global_max_requant,
    requantize_tensor,
    softmax_float,
)


ROOT = Path(
    r"D:\HCMUTE\HK I nam 4\DiDong\ECG_NAS_LOCAL"
)

STEM_TEST_DIR = (
    ROOT
    / "pynq"
    / "real_stem_test"
)

OUT_DIR = (
    ROOT
    / "pynq"
    / "full_inference_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


LABELS = [
    "N",
    "L",
    "R",
    "V",
    "A",
]


dep = DeploymentLoader(
    ROOT
)


print("=" * 72)
print("FULL ECG INTEGER INFERENCE")
print("=" * 72)


# ============================================================
# SOURCE ECG INFORMATION
# ============================================================

input_file = (
    STEM_TEST_DIR
    / "input_int8.npy"
)

config_file = (
    STEM_TEST_DIR
    / "config.json"
)


if not input_file.exists():
    raise FileNotFoundError(
        input_file
    )


ecg_input = np.load(
    input_file
).astype(
    np.int8
)


source_config = {}

if config_file.exists():

    with open(
        config_file,
        "r",
        encoding="utf-8"
    ) as f:

        source_config = json.load(
            f
        )


true_label = source_config.get(
    "label",
    "UNKNOWN"
)


print()
print("SOURCE")

print(
    "True label :",
    true_label
)

print(
    "Record ID  :",
    source_config.get(
        "record_id"
    )
)

print(
    "Input shape:",
    ecg_input.shape
)

print(
    "Input range:",
    int(ecg_input.min()),
    "...",
    int(ecg_input.max())
)


if ecg_input.shape != (
    320,
    1
):
    raise RuntimeError(
        f"ECG input shape sai: {ecg_input.shape}"
    )


# ============================================================
# STEM CONV
# ============================================================

stem_cfg = dep.layer(
    "stem_conv"
)

stem_kernel = dep.load_array(
    stem_cfg[
        "kernel_file"
    ],
    np.int8
)

stem_bias = dep.load_array(
    stem_cfg[
        "bias_file"
    ],
    np.int32
)

stem_req = stem_cfg[
    "acc_to_preact"
]


stem_preact = conv1d_same(
    x=ecg_input,

    kernel=stem_kernel,

    bias=stem_bias,

    multiplier=int(
        stem_req[
            "multiplier"
        ]
    ),

    shift=int(
        stem_req[
            "shift"
        ]
    ),

    output_bits=8,

    dilation=int(
        stem_cfg[
            "dilation"
        ]
    )
)


print()
print(
    "stem_preact:",
    stem_preact.shape,
    "range=",
    int(stem_preact.min()),
    "...",
    int(stem_preact.max())
)


# ============================================================
# OPTIONAL: VERIFY STEM AGAINST GOLDEN
# ============================================================

stem_golden_file = (
    STEM_TEST_DIR
    / "expected_int8.npy"
)


if stem_golden_file.exists():

    stem_golden = np.load(
        stem_golden_file
    ).astype(
        np.int8
    )

    stem_diff = (
        stem_preact.astype(
            np.int16
        )
        -
        stem_golden.astype(
            np.int16
        )
    )

    stem_mismatch = int(
        np.count_nonzero(
            stem_diff
        )
    )

    print(
        "stem mismatch:",
        stem_mismatch
    )

    if stem_mismatch != 0:
        raise RuntimeError(
            "Stem không còn bit-exact."
        )


# ============================================================
# BLOCK 1 -> POOL1
# ============================================================

block1_runtime = (
    RuntimeToPool1(
        ROOT
    )
)

block1 = block1_runtime.run(
    stem_preact
)

pool1 = block1[
    "pool1"
]


print()
print(
    "pool1:",
    pool1.shape,
    "range=",
    int(pool1.min()),
    "...",
    int(pool1.max())
)


# ============================================================
# BLOCK 2
# ============================================================

block_runtime = (
    ResidualInceptionBlock(
        dep
    )
)


block2 = block_runtime.run(
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
        f"pool2 shape sai: {pool2.shape}"
    )


# ============================================================
# BLOCK 3
# ============================================================

block3 = block_runtime.run(
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
        f"pool3 shape sai: {pool3.shape}"
    )


# ============================================================
# BLOCK 4
# ============================================================

block4 = block_runtime.run(
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
        f"block4_act shape sai: {block4_act.shape}"
    )


# ============================================================
# GAP + GMP
# ============================================================

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
    gap.shape
)

print(
    "gmp:",
    gmp.shape
)


# ============================================================
# HEAD CONCAT
# ============================================================

head_concat_scale = (
    dep.tensor_scale(
        "head_concat"
    )
)

head_concat_bits = (
    dep.tensor_bits(
        "head_concat"
    )
)


gap_aligned = requantize_tensor(
    q=gap,

    input_scale=
        dep.tensor_scale(
            "gap"
        ),

    output_scale=
        head_concat_scale,

    bits=
        head_concat_bits
)


gmp_aligned = requantize_tensor(
    q=gmp,

    input_scale=
        dep.tensor_scale(
            "gmp"
        ),

    output_scale=
        head_concat_scale,

    bits=
        head_concat_bits
)


head_concat = np.concatenate(
    [
        gap_aligned,
        gmp_aligned
    ]
).astype(
    np.int8
)


print(
    "head_concat:",
    head_concat.shape
)


if head_concat.shape != (
    56,
):
    raise RuntimeError(
        f"head_concat shape sai: "
        f"{head_concat.shape}"
    )


# ============================================================
# HEAD DENSE 56 -> 24
# ============================================================

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
        f"head_act shape sai: "
        f"{head_act.shape}"
    )


# ============================================================
# CLASS OUTPUT 24 -> 5
#
# output_mode = INT32_LOGITS
# Không requant về INT8.
# ============================================================

class_cfg = dep.layer(
    "class_output"
)


class_kernel = dep.load_array(
    class_cfg[
        "kernel_file"
    ],
    np.int8
)


class_bias = dep.load_array(
    class_cfg[
        "bias_file"
    ],
    np.int32
)


logits_int32 = dense_logits_int32(
    x=head_act,

    kernel=class_kernel,

    bias=class_bias
)


logit_scale = float(
    class_cfg[
        "logit_scale"
    ]
)


logits_float = (
    logits_int32.astype(
        np.float64
    )
    * logit_scale
)


probabilities = softmax_float(
    logits_float
)


predicted_id = int(
    np.argmax(
        logits_int32
    )
)


predicted_label = (
    LABELS[
        predicted_id
    ]
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 72)
print("CLASSIFICATION RESULT")
print("=" * 72)

print()
print(
    "INT32 logits:"
)

for i, label in enumerate(
    LABELS
):
    print(
        f"{label}: "
        f"{int(logits_int32[i]):8d}"
    )


print()
print(
    "Float logits:"
)

for i, label in enumerate(
    LABELS
):
    print(
        f"{label}: "
        f"{float(logits_float[i]): .6f}"
    )


print()
print(
    "Probabilities:"
)

for i, label in enumerate(
    LABELS
):
    print(
        f"{label}: "
        f"{float(probabilities[i]) * 100:6.2f}%"
    )


print()
print(
    "Predicted class:",
    predicted_label
)

print(
    "True label     :",
    true_label
)


if true_label in LABELS:

    correct = (
        predicted_label
        == true_label
    )

    print(
        "Correct        :",
        correct
    )


# ============================================================
# SAVE RESULT
# ============================================================

np.save(
    OUT_DIR
    / "logits_int32.npy",
    logits_int32
)

np.save(
    OUT_DIR
    / "logits_float.npy",
    logits_float
)

np.save(
    OUT_DIR
    / "probabilities.npy",
    probabilities
)


result_json = {
    "record_id":
        source_config.get(
            "record_id"
        ),

    "true_label":
        true_label,

    "predicted_id":
        predicted_id,

    "predicted_label":
        predicted_label,

    "logit_scale":
        logit_scale,

    "logits_int32":
        [
            int(x)
            for x in logits_int32
        ],

    "logits_float":
        [
            float(x)
            for x in logits_float
        ],

    "probabilities":
        [
            float(x)
            for x in probabilities
        ],
}


with open(
    OUT_DIR
    / "result.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result_json,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print(
    "Saved result:"
)

print(
    OUT_DIR
    / "result.json"
)


print()
print("=" * 72)
print("PASS: FULL INTEGER ECG INFERENCE COMPLETED")
print("=" * 72)