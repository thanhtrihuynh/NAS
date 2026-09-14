from pathlib import Path

import numpy as np

from deployment_loader import DeploymentLoader

from integer_ops import (
    conv1d_same,
    gelu_requant,
    maxpool1d_same3_s1,
    maxpool1d_2_s2,
    requantize_scale,
    residual_add_scaled,
    sepconv_preact,
)


class RuntimeToPool1:

    def __init__(self, project_root):
        self.root = Path(project_root)

        self.dep = DeploymentLoader(
            self.root
        )

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

        (
            depthwise_out,
            preactivation
        ) = sepconv_preact(
            x=x,
            cfg=cfg,
            depthwise=depthwise,
            pointwise=pointwise,
            bias=bias
        )

        activation = gelu_requant(
            q_in=preactivation,
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
            "depthwise":
                depthwise_out,

            "preact":
                preactivation,

            "act":
                activation,

            "preact_scale":
                float(
                    cfg[
                        "preactivation_scale"
                    ]
                ),

            "act_scale":
                float(
                    cfg[
                        "output_scale"
                    ]
                ),
        }

    def run_pool_branch(
        self,
        stem_act
    ):
        pool = maxpool1d_same3_s1(
            stem_act
        )

        (
            cfg,
            kernel,
            bias
        ) = self.dep.load_conv(
            "block1_pool_proj"
        )

        req = cfg[
            "acc_to_preact"
        ]

        preactivation = conv1d_same(
            x=pool,
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

        activation = gelu_requant(
            q_in=preactivation,
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
            "pool":
                pool,

            "preact":
                preactivation,

            "act":
                activation,

            "act_scale":
                float(
                    cfg[
                        "output_scale"
                    ]
                ),
        }

    def run_out_projection(
        self,
        concat_q
    ):
        (
            cfg,
            kernel,
            bias
        ) = self.dep.load_conv(
            "block1_out_proj"
        )

        req = cfg[
            "acc_to_output"
        ]

        output = conv1d_same(
            x=concat_q,
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
            "output":
                output,

            "scale":
                float(
                    cfg[
                        "output_scale"
                    ]
                ),
        }

    def run(self, stem_preact):
        stem_preact = np.asarray(
            stem_preact,
            dtype=np.int8
        )

        if stem_preact.shape != (
            320,
            8
        ):
            raise ValueError(
                "stem_preact phải có "
                f"shape (320,8), nhận "
                f"{stem_preact.shape}"
            )

        stem_cfg = self.dep.layer(
            "stem_conv"
        )

        stem_preact_scale = float(
            stem_cfg[
                "preactivation_scale"
            ]
        )

        stem_act_scale = float(
            stem_cfg[
                "output_scale"
            ]
        )

        stem_act = gelu_requant(
            q_in=stem_preact,
            input_scale=stem_preact_scale,
            output_scale=stem_act_scale,
            output_bits=8
        )

        print(
            "stem_act:",
            stem_act.shape
        )

        b1 = self.run_sepconv(
            "block1_b1_sepconv",
            stem_act
        )

        b2 = self.run_sepconv(
            "block1_b2_sepconv",
            stem_act
        )

        b3 = self.run_sepconv(
            "block1_b3_sepconv",
            stem_act
        )

        pool_branch = (
            self.run_pool_branch(
                stem_act
            )
        )

        print(
            "block1_b1_act:",
            b1["act"].shape
        )

        print(
            "block1_b2_act:",
            b2["act"].shape
        )

        print(
            "block1_b3_act:",
            b3["act"].shape
        )

        print(
            "block1_pool_act:",
            pool_branch[
                "act"
            ].shape
        )

        concat_scale = (
            self.dep.tensor_scale(
                "block1_concat"
            )
        )

        concat_bits = (
            self.dep.tensor_bits(
                "block1_concat"
            )
        )

        b1_aligned = requantize_scale(
            b1["act"],
            b1["act_scale"],
            concat_scale,
            concat_bits
        )

        b2_aligned = requantize_scale(
            b2["act"],
            b2["act_scale"],
            concat_scale,
            concat_bits
        )

        b3_aligned = requantize_scale(
            b3["act"],
            b3["act_scale"],
            concat_scale,
            concat_bits
        )

        pool_aligned = requantize_scale(
            pool_branch["act"],
            pool_branch["act_scale"],
            concat_scale,
            concat_bits
        )

        block1_concat = np.concatenate(
            [
                b1_aligned,
                b2_aligned,
                b3_aligned,
                pool_aligned,
            ],
            axis=1
        ).astype(np.int8)

        if block1_concat.shape != (
            320,
            16
        ):
            raise RuntimeError(
                "Sai shape block1_concat: "
                f"{block1_concat.shape}"
            )

        print(
            "block1_concat:",
            block1_concat.shape
        )

        out_proj = (
            self.run_out_projection(
                block1_concat
            )
        )

        block1_out_bn = (
            out_proj["output"]
        )

        block1_out_bn_scale = (
            out_proj["scale"]
        )

        print(
            "block1_out_bn:",
            block1_out_bn.shape
        )

        add_scale = (
            self.dep.tensor_scale(
                "block1_add"
            )
        )

        add_bits = (
            self.dep.tensor_bits(
                "block1_add"
            )
        )

        block1_add = (
            residual_add_scaled(
                main_q=block1_out_bn,
                main_scale=block1_out_bn_scale,
                shortcut_q=stem_act,
                shortcut_scale=stem_act_scale,
                output_scale=add_scale,
                output_bits=add_bits
            )
        )

        print(
            "block1_add:",
            block1_add.shape
        )

        block1_act_scale = (
            self.dep.tensor_scale(
                "block1_act"
            )
        )

        block1_act_bits = (
            self.dep.tensor_bits(
                "block1_act"
            )
        )

        block1_act = gelu_requant(
            q_in=block1_add,
            input_scale=add_scale,
            output_scale=block1_act_scale,
            output_bits=block1_act_bits
        )

        print(
            "block1_act:",
            block1_act.shape
        )

        pool1 = maxpool1d_2_s2(
            block1_act
        )

        print(
            "pool1:",
            pool1.shape
        )

        if pool1.shape != (
            160,
            8
        ):
            raise RuntimeError(
                "Sai shape pool1: "
                f"{pool1.shape}"
            )

        pool1_scale = (
            self.dep.tensor_scale(
                "pool1"
            )
        )

        return {
            "stem_act":
                stem_act,

            "block1_b1_depthwise":
                b1["depthwise"],

            "block1_b1_preact":
                b1["preact"],

            "block1_b1_act":
                b1["act"],

            "block1_b2_depthwise":
                b2["depthwise"],

            "block1_b2_preact":
                b2["preact"],

            "block1_b2_act":
                b2["act"],

            "block1_b3_depthwise":
                b3["depthwise"],

            "block1_b3_preact":
                b3["preact"],

            "block1_b3_act":
                b3["act"],

            "block1_pool":
                pool_branch["pool"],

            "block1_pool_preact":
                pool_branch["preact"],

            "block1_pool_act":
                pool_branch["act"],

            "block1_concat":
                block1_concat,

            "block1_out_bn":
                block1_out_bn,

            "block1_add":
                block1_add,

            "block1_act":
                block1_act,

            "pool1":
                pool1,

            "pool1_scale":
                pool1_scale,
        }