import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers

from config import NUM_CLASSES, SEGMENT_LEN

L2_LAMBDA = 5e-5


def _sep_branch(x, filters, kernel_size, dilation_rate, name):
    x = layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(L2_LAMBDA),
        pointwise_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{name}_sepconv",
    )(x)

    x = layers.BatchNormalization(
        epsilon=1e-5,
        name=f"{name}_bn",
    )(x)

    return layers.Activation(
        tf.nn.gelu,
        name=f"{name}_act",
    )(x)


def _inception_block(x, kernels, filters, block_name, dropout=0.05):
    shortcut = x
    branch_filters = max(filters // 2, 4)

    b1 = _sep_branch(
        x, branch_filters, kernels[0], 1, f"{block_name}_b1"
    )
    b2 = _sep_branch(
        x, branch_filters, kernels[1], 1, f"{block_name}_b2"
    )
    b3 = _sep_branch(
        x, branch_filters, kernels[2], 2, f"{block_name}_b3"
    )

    b4 = layers.MaxPooling1D(
        pool_size=3,
        strides=1,
        padding="same",
        name=f"{block_name}_b4_pool",
    )(x)

    b4 = layers.Conv1D(
        branch_filters,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{block_name}_b4_conv",
    )(b4)

    b4 = layers.BatchNormalization(
        epsilon=1e-5,
        name=f"{block_name}_b4_bn",
    )(b4)

    b4 = layers.Activation(
        tf.nn.gelu,
        name=f"{block_name}_b4_act",
    )(b4)

    x = layers.Concatenate(
        axis=-1,
        name=f"{block_name}_concat",
    )([b1, b2, b3, b4])

    x = layers.Conv1D(
        filters,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{block_name}_out_proj",
    )(x)

    x = layers.BatchNormalization(
        epsilon=1e-5,
        name=f"{block_name}_out_bn",
    )(x)

    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv1D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(L2_LAMBDA),
            name=f"{block_name}_shortcut_conv",
        )(shortcut)

        shortcut = layers.BatchNormalization(
            epsilon=1e-5,
            name=f"{block_name}_shortcut_bn",
        )(shortcut)

    x = layers.Add(
        name=f"{block_name}_add"
    )([x, shortcut])

    x = layers.Activation(
        tf.nn.gelu,
        name=f"{block_name}_out_act",
    )(x)

    return layers.SpatialDropout1D(
        dropout,
        name=f"{block_name}_dropout",
    )(x)


def build_nas_model(
    architecture,
    input_len=SEGMENT_LEN,
    num_classes=NUM_CLASSES,
):
    inputs = layers.Input(
        shape=(input_len, 1),
        name="ecg_input",
    )

    x = layers.Conv1D(
        8,
        7,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="stem_conv",
    )(inputs)

    x = layers.BatchNormalization(
        epsilon=1e-5,
        name="stem_bn",
    )(x)

    x = layers.Activation(
        tf.nn.gelu,
        name="stem_act",
    )(x)

    for i in range(1, 5):
        x = _inception_block(
            x,
            architecture[f"block{i}_kernels"],
            int(architecture[f"block{i}_channels"]),
            f"block{i}",
        )

        if i < 4:
            x = layers.MaxPooling1D(
                pool_size=2,
                strides=2,
                name=f"pool{i}",
            )(x)

    gap = layers.GlobalAveragePooling1D(name="gap")(x)
    gmp = layers.GlobalMaxPooling1D(name="gmp")(x)

    x = layers.Concatenate(
        name="global_concat"
    )([gap, gmp])

    x = layers.Dense(
        24,
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="head_dense",
    )(x)

    x = layers.BatchNormalization(
        epsilon=1e-5,
        name="head_bn",
    )(x)

    x = layers.Activation(
        tf.nn.gelu,
        name="head_act",
    )(x)

    x = layers.Dropout(
        0.20,
        name="head_dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(x)

    return Model(
        inputs,
        outputs,
        name="NAS_ECG_MiniInception",
    )
