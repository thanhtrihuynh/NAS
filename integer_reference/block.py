from integer_reference.ops import (
    requantize_from_scale
)

from integer_reference.pooling import (
    max_pool1d_integer
)

from integer_reference.merge_ops import (
    concat_integer,
    residual_add_integer
)

from integer_reference.activation import (
    gelu_integer_lut
)


def get_tensor_info(
    manifest,
    tensor_name
):
    scales = manifest[
        "tensor_scales"
    ]

    if tensor_name not in scales:
        raise KeyError(
            "Không tìm thấy tensor scale: "
            f"{tensor_name}"
        )

    info = scales[
        tensor_name
    ]

    return (
        float(
            info["scale"]
        ),
        int(
            info["bits"]
        )
    )


def run_weight_layer_output(
    runner,
    manifest,
    layer_name,
    x_q
):
    info = manifest[
        "layers"
    ][layer_name]

    (
        accumulator,
        accumulator_scale
    ) = runner.run_accumulator(
        layer_name,
        x_q
    )

    output_mode = info[
        "output_mode"
    ]

    if output_mode == "REQUANT":
        output_scale = float(
            info[
                "output_scale"
            ]
        )

        output_bits = int(
            info[
                "output_bits"
            ]
        )

        output_q = requantize_from_scale(
            accumulator,
            accumulator_scale,
            output_scale,
            output_bits
        )

        return (
            output_q,
            output_scale,
            output_bits
        )

    if (
        output_mode
        == "REQUANT_GELU_REQUANT"
    ):
        preactivation_scale = float(
            info[
                "preactivation_scale"
            ]
        )

        bn_name = info.get(
            "bn_name"
        )

        if (
            bn_name is not None
            and bn_name
            in manifest[
                "tensor_scales"
            ]
        ):
            preactivation_bits = int(
                manifest[
                    "tensor_scales"
                ][
                    bn_name
                ][
                    "bits"
                ]
            )

        else:
            preactivation_bits = 8

        preactivation_q = (
            requantize_from_scale(
                accumulator,
                accumulator_scale,
                preactivation_scale,
                preactivation_bits
            )
        )

        output_scale = float(
            info[
                "output_scale"
            ]
        )

        output_bits = int(
            info[
                "output_bits"
            ]
        )

        output_q = gelu_integer_lut(
            preactivation_q,
            preactivation_scale,
            preactivation_bits,
            output_scale,
            output_bits
        )

        return (
            output_q,
            output_scale,
            output_bits
        )

    raise ValueError(
        "Output mode chưa hỗ trợ cho "
        f"{layer_name}: "
        f"{output_mode}"
    )


class IntegerResidualBlock:
    def __init__(
        self,
        block_index,
        runner,
        manifest
    ):
        self.block_index = int(
            block_index
        )

        self.runner = runner

        self.manifest = manifest

        self.prefix = (
            f"block{self.block_index}"
        )

    def forward(
        self,
        x_q,
        input_scale,
        input_bits,
        trace=None
    ):
        prefix = self.prefix

        # ==========================================
        # Branch 1
        # SepConv -> BN folded -> GELU
        # ==========================================

        (
            b1_q,
            b1_scale,
            b1_bits
        ) = run_weight_layer_output(
            self.runner,
            self.manifest,
            f"{prefix}_b1_sepconv",
            x_q
        )

        # ==========================================
        # Branch 2
        # ==========================================

        (
            b2_q,
            b2_scale,
            b2_bits
        ) = run_weight_layer_output(
            self.runner,
            self.manifest,
            f"{prefix}_b2_sepconv",
            x_q
        )

        # ==========================================
        # Branch 3
        # ==========================================

        (
            b3_q,
            b3_scale,
            b3_bits
        ) = run_weight_layer_output(
            self.runner,
            self.manifest,
            f"{prefix}_b3_sepconv",
            x_q
        )

        # ==========================================
        # Pool branch
        #
        # MaxPool3 stride1 same
        # -> 1x1 Conv
        # -> BN folded
        # -> GELU
        # ==========================================

        pool_q = max_pool1d_integer(
            x_q,
            pool_size=3,
            stride=1,
            padding="same"
        )

        (
            pool_proj_q,
            pool_proj_scale,
            pool_proj_bits
        ) = run_weight_layer_output(
            self.runner,
            self.manifest,
            f"{prefix}_pool_proj",
            pool_q
        )

        # ==========================================
        # Concat
        # ==========================================

        concat_name = (
            f"{prefix}_concat"
        )

        (
            concat_scale,
            concat_bits
        ) = get_tensor_info(
            self.manifest,
            concat_name
        )

        concat_q = concat_integer(
            [
                b1_q,
                b2_q,
                b3_q,
                pool_proj_q
            ],
            [
                b1_scale,
                b2_scale,
                b3_scale,
                pool_proj_scale
            ],
            concat_scale,
            concat_bits,
            axis=-1
        )

        # ==========================================
        # Output projection
        #
        # Conv1x1 -> BN folded
        # ==========================================

        (
            main_q,
            main_scale,
            main_bits
        ) = run_weight_layer_output(
            self.runner,
            self.manifest,
            f"{prefix}_out_proj",
            concat_q
        )

        # ==========================================
        # Shortcut
        # ==========================================

        shortcut_layer_name = (
            f"{prefix}_shortcut"
        )

        has_projection_shortcut = (
            shortcut_layer_name
            in self.manifest[
                "layers"
            ]
        )

        if has_projection_shortcut:
            (
                shortcut_q,
                shortcut_scale,
                shortcut_bits
            ) = run_weight_layer_output(
                self.runner,
                self.manifest,
                shortcut_layer_name,
                x_q
            )

        else:
            shortcut_q = x_q

            shortcut_scale = float(
                input_scale
            )

            shortcut_bits = int(
                input_bits
            )

        # ==========================================
        # Residual Add
        # ==========================================

        add_name = (
            f"{prefix}_add"
        )

        (
            add_scale,
            add_bits
        ) = get_tensor_info(
            self.manifest,
            add_name
        )

        add_q = residual_add_integer(
            main_q,
            main_scale,
            shortcut_q,
            shortcut_scale,
            add_scale,
            add_bits
        )

        # ==========================================
        # GELU sau residual
        # ==========================================

        act_name = (
            f"{prefix}_act"
        )

        (
            act_scale,
            act_bits
        ) = get_tensor_info(
            self.manifest,
            act_name
        )

        output_q = gelu_integer_lut(
            add_q,
            add_scale,
            add_bits,
            act_scale,
            act_bits
        )

        # ==========================================
        # Trace dùng đúng tên semantic của QAT graph
        # ==========================================

        if trace is not None:
            trace[
                f"{prefix}_b1_act"
            ] = {
                "tensor":
                    b1_q,

                "scale":
                    b1_scale
            }

            trace[
                f"{prefix}_b2_act"
            ] = {
                "tensor":
                    b2_q,

                "scale":
                    b2_scale
            }

            trace[
                f"{prefix}_b3_act"
            ] = {
                "tensor":
                    b3_q,

                "scale":
                    b3_scale
            }

            trace[
                f"{prefix}_pool_act"
            ] = {
                "tensor":
                    pool_proj_q,

                "scale":
                    pool_proj_scale
            }

            trace[
                concat_name
            ] = {
                "tensor":
                    concat_q,

                "scale":
                    concat_scale
            }

            trace[
                f"{prefix}_out_bn"
            ] = {
                "tensor":
                    main_q,

                "scale":
                    main_scale
            }

            if has_projection_shortcut:
                trace[
                    f"{prefix}_shortcut_bn"
                ] = {
                    "tensor":
                        shortcut_q,

                    "scale":
                        shortcut_scale
                }

            trace[
                add_name
            ] = {
                "tensor":
                    add_q,

                "scale":
                    add_scale
            }

            trace[
                act_name
            ] = {
                "tensor":
                    output_q,

                "scale":
                    act_scale
            }

        return (
            output_q,
            act_scale,
            act_bits
        )