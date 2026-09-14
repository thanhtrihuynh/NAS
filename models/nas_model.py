from tensorflow.keras import layers, regularizers, Model

from config import (
    SEGMENT_LEN,
    NUM_CLASSES,
    L2_LAMBDA,
    DROPOUT_1,
    DROPOUT_2,
    DROPOUT_HEAD,
)


def nas_sep_conv_branch(
    x,
    filters,
    kernel_size,
    dilation_rate=1,
    name="branch",
):
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

    y = layers.BatchNormalization(name=f"{name}_bn")(y)
    y = layers.Activation("gelu", name=f"{name}_act")(y)
    return y


def nas_inception_block(x, filters, kernels, dropout_rate, block_name):
    shortcut = x
    branch_filters = max(filters // 2, 4)
    k1, k2, k3 = kernels

    b1 = nas_sep_conv_branch(
        x,
        branch_filters,
        kernel_size=k1,
        dilation_rate=1,
        name=f"{block_name}_b1",
    )

    b2 = nas_sep_conv_branch(
        x,
        branch_filters,
        kernel_size=k2,
        dilation_rate=1,
        name=f"{block_name}_b2",
    )

    b3 = nas_sep_conv_branch(
        x,
        branch_filters,
        kernel_size=k3,
        dilation_rate=2,
        name=f"{block_name}_b3",
    )

    b4 = layers.MaxPooling1D(
        pool_size=3,
        strides=1,
        padding="same",
        name=f"{block_name}_pool",
    )(x)

    b4 = layers.Conv1D(
        branch_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{block_name}_pool_proj",
    )(b4)
    b4 = layers.BatchNormalization(name=f"{block_name}_pool_bn")(b4)
    b4 = layers.Activation("gelu", name=f"{block_name}_pool_act")(b4)

    y = layers.Concatenate(name=f"{block_name}_concat")([b1, b2, b3, b4])

    y = layers.Conv1D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name=f"{block_name}_out_proj",
    )(y)
    y = layers.BatchNormalization(name=f"{block_name}_out_bn")(y)

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
    y = layers.Activation("gelu", name=f"{block_name}_act")(y)
    y = layers.SpatialDropout1D(dropout_rate, name=f"{block_name}_drop")(y)

    return y


def build_nas_model(config, input_shape=(SEGMENT_LEN, 1)):
    inp = layers.Input(shape=input_shape, name="ecg_input")

    x = layers.Conv1D(
        8,
        kernel_size=7,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="stem_conv",
    )(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("gelu", name="stem_act")(x)

    x = nas_inception_block(
        x,
        filters=config["block1_channels"],
        kernels=config["block1_kernels"],
        dropout_rate=DROPOUT_1,
        block_name="block1",
    )
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    x = nas_inception_block(
        x,
        filters=config["block2_channels"],
        kernels=config["block2_kernels"],
        dropout_rate=DROPOUT_1,
        block_name="block2",
    )
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    x = nas_inception_block(
        x,
        filters=config["block3_channels"],
        kernels=config["block3_kernels"],
        dropout_rate=DROPOUT_2,
        block_name="block3",
    )
    x = layers.MaxPooling1D(pool_size=2, name="pool3")(x)

    x = nas_inception_block(
        x,
        filters=config["block4_channels"],
        kernels=config["block4_kernels"],
        dropout_rate=DROPOUT_2,
        block_name="block4",
    )

    gap = layers.GlobalAveragePooling1D(name="gap")(x)
    gmp = layers.GlobalMaxPooling1D(name="gmp")(x)
    x = layers.Concatenate(name="head_concat")([gap, gmp])

    x = layers.Dense(
        24,
        use_bias=False,
        kernel_regularizer=regularizers.l2(L2_LAMBDA),
        name="head_dense",
    )(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation("gelu", name="head_act")(x)
    x = layers.Dropout(DROPOUT_HEAD, name="head_dropout")(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax", name="class_output")(x)

    return Model(inp, out, name="nas_inception")
