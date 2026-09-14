import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

from config import (
    SEGMENT_LEN,
    NUM_CLASSES,
    L2_LAMBDA,
    DROPOUT_1,
    DROPOUT_2,
    DROPOUT_HEAD,
)


def bn_gelu(x, name):
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    return x


def sep_conv_branch(x, filters, kernel_size, dilation_rate, name):
    y = layers.SeparableConv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        depthwise_regularizer=regularizers.l2(L2_LAMBDA),
        pointwise_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{name}_sepconv",
    )(x)

    return bn_gelu(y, name=name)


def sep_res_inception_block(x, filters, dropout_rate, block_name):
    shortcut = x
    branch_filters = max(filters // 2, 4)

    b1 = sep_conv_branch(
        x,
        filters=branch_filters,
        kernel_size=3,
        dilation_rate=1,
        name=f"{block_name}_b1_k3",
    )

    b2 = sep_conv_branch(
        x,
        filters=branch_filters,
        kernel_size=5,
        dilation_rate=1,
        name=f"{block_name}_b2_k5",
    )

    b3 = sep_conv_branch(
        x,
        filters=branch_filters,
        kernel_size=7,
        dilation_rate=2,
        name=f"{block_name}_b3_k7_d2",
    )

    b4 = layers.MaxPooling1D(
        pool_size=3,
        strides=1,
        padding="same",
        name=f"{block_name}_b4_pool",
    )(x)

    b4 = layers.Conv1D(
        branch_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{block_name}_b4_conv1x1",
    )(b4)

    b4 = bn_gelu(b4, name=f"{block_name}_b4")

    y = layers.Concatenate(name=f"{block_name}_concat")([b1, b2, b3, b4])

    y = layers.Conv1D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{block_name}_projection",
    )(y)

    y = layers.BatchNormalization(name=f"{block_name}_projection_bn")(y)

    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(L2_LAMBDA),
            name=f"{block_name}_shortcut",
        )(shortcut)

        shortcut = layers.BatchNormalization(name=f"{block_name}_shortcut_bn")(
            shortcut
        )

    y = layers.Add(name=f"{block_name}_add")([shortcut, y])
    y = layers.Activation("gelu", name=f"{block_name}_output_gelu")(y)
    y = layers.SpatialDropout1D(dropout_rate, name=f"{block_name}_output")(y)

    return y


def build_baseline_model(input_shape=(SEGMENT_LEN, 1)):
    inp = layers.Input(shape=input_shape, name="ecg_input")

    x = layers.Conv1D(
        filters=8,
        kernel_size=7,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="stem_conv_k7",
    )(inp)
    x = bn_gelu(x, name="stem")

    x = sep_res_inception_block(
        x, filters=8, dropout_rate=DROPOUT_1, block_name="block1"
    )
    x = layers.MaxPooling1D(pool_size=2, name="downsample1")(x)

    x = sep_res_inception_block(
        x, filters=12, dropout_rate=DROPOUT_1, block_name="block2"
    )
    x = layers.MaxPooling1D(pool_size=2, name="downsample2")(x)

    x = sep_res_inception_block(
        x, filters=16, dropout_rate=DROPOUT_2, block_name="block3"
    )
    x = layers.MaxPooling1D(pool_size=2, name="downsample3")(x)

    x = sep_res_inception_block(
        x, filters=20, dropout_rate=DROPOUT_2, block_name="block4"
    )

    gap = layers.GlobalAveragePooling1D(name="head_global_avg_pool")(x)
    gmp = layers.GlobalMaxPooling1D(name="head_global_max_pool")(x)
    x = layers.Concatenate(name="head_concat")([gap, gmp])

    x = layers.Dense(
        24,
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("gelu", name="head_gelu")(x)
    x = layers.Dropout(DROPOUT_HEAD, name="head_dropout")(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)

    return Model(inp, out, name="sep_res_inception_gelu_5class")
