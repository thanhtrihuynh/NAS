import numpy as np

from integer_ops import (
    conv1d_same,
    gelu_requant,
    maxpool1d_same3_s1,
    maxpool1d_2_s2,
    requantize_tensor,
    residual_add_scaled,
    sepconv_preact_mixed,
)


class ResidualInceptionBlock:

    def __init__(
        self,
        deployment_loader
    ):
        self.dep = deployment_loader


    def run_sepconv(
        self,
        layer_name,
        x
    ):
        (
            cfg,
            depthwise,
            pointwise,
            bias
        ) = self.dep.load_sepconv(
            layer_name
        )

        preact_bits = (
            self.dep.tensor_bits(
                cfg["bn_name"]
            )
        )

        dw, preact = (
            sepconv_preact_mixed(
                x=x,
                cfg=cfg,
                depthwise=depthwise,
                pointwise=pointwise,
                bias=bias,
                preact_bits=preact_bits
            )
        )

        act = gelu_requant(
            q_in=preact,
            input_scale=float(
                cfg[
                    "preactivation_scale"
                ]
            ),
            output_scale=float(
                cfg[
                    "output_scale"
                ]
            ),
            output_bits=int(
                cfg[
                    "output_bits"
                ]
            )
        )

        return {
            "depthwise": dw,
            "preact": preact,
            "act": act,
            "scale": float(
                cfg["output_scale"]
            ),
            "bits": int(
                cfg["output_bits"]
            ),
        }


    def run_pool_branch(
        self,
        prefix,
        x
    ):
        pool = maxpool1d_same3_s1(
            x
        )

        layer_name = (
            f"{prefix}_pool_proj"
        )

        (
            cfg,
            kernel,
            bias
        ) = self.dep.load_conv(
            layer_name
        )

        req = cfg[
            "acc_to_preact"
        ]

        preact_bits = (
            self.dep.tensor_bits(
                cfg["bn_name"]
            )
        )

        preact = conv1d_same(
            x=pool,
            kernel=kernel,
            bias=bias,
            multiplier=int(
                req["multiplier"]
            ),
            shift=int(
                req["shift"]
            ),
            output_bits=preact_bits,
            dilation=1
        )

        act = gelu_requant(
            q_in=preact,
            input_scale=float(
                cfg[
                    "preactivation_scale"
                ]
            ),
            output_scale=float(
                cfg[
                    "output_scale"
                ]
            ),
            output_bits=int(
                cfg[
                    "output_bits"
                ]
            )
        )

        return {
            "pool": pool,
            "preact": preact,
            "act": act,
            "scale": float(
                cfg["output_scale"]
            ),
        }


    def run_linear_conv(
        self,
        layer_name,
        x
    ):
        (
            cfg,
            kernel,
            bias
        ) = self.dep.load_conv(
            layer_name
        )

        req = cfg[
            "acc_to_output"
        ]

        output = conv1d_same(
            x=x,
            kernel=kernel,
            bias=bias,
            multiplier=int(
                req["multiplier"]
            ),
            shift=int(
                req["shift"]
            ),
            output_bits=int(
                cfg["output_bits"]
            ),
            dilation=1
        )

        return {
            "output": output,
            "scale": float(
                cfg["output_scale"]
            ),
        }


    def run(
        self,
        prefix,
        x,
        input_scale,
        downsample
    ):
        b1 = self.run_sepconv(
            f"{prefix}_b1_sepconv",
            x
        )

        b2 = self.run_sepconv(
            f"{prefix}_b2_sepconv",
            x
        )

        b3 = self.run_sepconv(
            f"{prefix}_b3_sepconv",
            x
        )

        pool_branch = (
            self.run_pool_branch(
                prefix,
                x
            )
        )

        concat_name = (
            f"{prefix}_concat"
        )

        concat_scale = (
            self.dep.tensor_scale(
                concat_name
            )
        )

        concat_bits = (
            self.dep.tensor_bits(
                concat_name
            )
        )

        branches = []

        for branch in [
            b1,
            b2,
            b3,
            pool_branch,
        ]:
            aligned = requantize_tensor(
                branch["act"],
                branch["scale"],
                concat_scale,
                concat_bits
            )

            branches.append(
                aligned
            )

        concat = np.concatenate(
            branches,
            axis=1
        ).astype(
            np.int8
        )

        out_proj = (
            self.run_linear_conv(
                f"{prefix}_out_proj",
                concat
            )
        )

        shortcut = (
            self.run_linear_conv(
                f"{prefix}_shortcut",
                x
            )
        )

        add_scale = (
            self.dep.tensor_scale(
                f"{prefix}_add"
            )
        )

        add_bits = (
            self.dep.tensor_bits(
                f"{prefix}_add"
            )
        )

        add = residual_add_scaled(
            main_q=
                out_proj["output"],

            main_scale=
                out_proj["scale"],

            shortcut_q=
                shortcut["output"],

            shortcut_scale=
                shortcut["scale"],

            output_scale=
                add_scale,

            output_bits=
                add_bits
        )

        act_scale = (
            self.dep.tensor_scale(
                f"{prefix}_act"
            )
        )

        act_bits = (
            self.dep.tensor_bits(
                f"{prefix}_act"
            )
        )

        act = gelu_requant(
            q_in=add,
            input_scale=add_scale,
            output_scale=act_scale,
            output_bits=act_bits
        )

        if downsample:

            pooled = (
                maxpool1d_2_s2(
                    act
                )
            )

        else:

            pooled = None

        return {
            "b1": b1,
            "b2": b2,
            "b3": b3,
            "pool_branch":
                pool_branch,

            "concat":
                concat,

            "out_proj":
                out_proj[
                    "output"
                ],

            "shortcut":
                shortcut[
                    "output"
                ],

            "add":
                add,

            "act":
                act,

            "act_scale":
                act_scale,

            "pooled":
                pooled,
        }