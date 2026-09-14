import time

import numpy as np

from deployment_loader import (
    DeploymentLoader,
)
from fpga_backend import (
    FpgaLayerExecutor,
)
from integer_nonweighted import (
    dense_logits_int32,
    gelu_requant,
    global_average_requant,
    global_max_requant,
    maxpool1d_2_s2,
    maxpool1d_same3_s1,
    quantize,
    requantize_tensor,
    residual_add_scaled,
    softmax_float,
)


LABELS = [
    "N",
    "L",
    "R",
    "V",
    "A",
]


class HybridFpgaRuntime:
    def __init__(
        self,
        deployment_root,
        bitstream=None,
        dma_name=None,
    ):
        self.dep = DeploymentLoader(
            deployment_root
        )

        self.fpga = FpgaLayerExecutor(
            bitstream=bitstream,
            dma_name=dma_name,
        )

    def _conv(
        self,
        layer_name,
        x,
        request_key,
        output_bits=None,
    ):
        cfg = self.dep.layer(
            layer_name
        )

        kernel = self.dep.load_array(
            cfg["kernel_file"],
            np.int8,
        )

        bias = self.dep.load_array(
            cfg["bias_file"],
            np.int32,
        )

        req = cfg[
            request_key
        ]

        if output_bits is None:
            output_bits = int(
                cfg["output_bits"]
            )

        y, ms = self.fpga.execute(
            x=x,
            weights=kernel,
            bias=bias,
            multiplier=int(
                req["multiplier"]
            ),
            shift=int(
                req["shift"]
            ),
            output_bits=int(
                output_bits
            ),
            dilation=int(
                cfg.get(
                    "dilation",
                    1,
                )
            ),
            depthwise=False,
            weight_bits=int(
                cfg.get(
                    "bits",
                    8,
                )
            ),
        )

        return y, ms

    def _sepconv(
        self,
        layer_name,
        x,
    ):
        cfg = self.dep.layer(
            layer_name
        )

        depthwise = self.dep.load_array(
            cfg["depthwise_file"],
            np.int8,
        )

        pointwise = self.dep.load_array(
            cfg["pointwise_file"],
            np.int8,
        )

        bias = self.dep.load_array(
            cfg["bias_file"],
            np.int32,
        )

        bits = int(
            cfg["bits"]
        )

        zeros = np.zeros(
            depthwise.shape[1],
            dtype=np.int32,
        )

        dw_req = cfg[
            "depthwise_requant"
        ]

        dw, dw_ms = self.fpga.execute(
            x=x,
            weights=depthwise,
            bias=zeros,
            multiplier=int(
                dw_req["multiplier"]
            ),
            shift=int(
                dw_req["shift"]
            ),
            output_bits=bits,
            dilation=int(
                cfg.get(
                    "dilation",
                    1,
                )
            ),
            depthwise=True,
            weight_bits=bits,
        )

        preact_bits = (
            self.dep.tensor_bits(
                cfg["bn_name"]
            )
            if cfg.get("bn_name")
            else 8
        )

        pw_req = cfg[
            "acc_to_preact"
        ]

        preact, pw_ms = (
            self.fpga.execute(
                x=dw,
                weights=pointwise,
                bias=bias,
                multiplier=int(
                    pw_req["multiplier"]
                ),
                shift=int(
                    pw_req["shift"]
                ),
                output_bits=preact_bits,
                dilation=1,
                depthwise=False,
                weight_bits=bits,
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
            ),
        )

        return {
            "act": act,
            "scale": float(
                cfg["output_scale"]
            ),
            "fpga_ms":
                dw_ms
                + pw_ms,
        }

    def _block(
        self,
        prefix,
        x,
        input_scale,
        downsample,
    ):
        fpga_ms = 0.0

        branches = []

        for branch_name in [
            "b1",
            "b2",
            "b3",
        ]:
            result = self._sepconv(
                f"{prefix}_{branch_name}_sepconv",
                x,
            )

            branches.append(
                result
            )

            fpga_ms += result[
                "fpga_ms"
            ]

        pool_x = maxpool1d_same3_s1(
            x
        )

        pool_cfg = self.dep.layer(
            f"{prefix}_pool_proj"
        )

        pool_preact_bits = (
            self.dep.tensor_bits(
                pool_cfg["bn_name"]
            )
        )

        pool_preact, ms = self._conv(
            f"{prefix}_pool_proj",
            pool_x,
            "acc_to_preact",
            output_bits=
                pool_preact_bits,
        )

        fpga_ms += ms

        pool_act = gelu_requant(
            q_in=pool_preact,
            input_scale=float(
                pool_cfg[
                    "preactivation_scale"
                ]
            ),
            output_scale=float(
                pool_cfg[
                    "output_scale"
                ]
            ),
            output_bits=int(
                pool_cfg[
                    "output_bits"
                ]
            ),
        )

        concat_scale = (
            self.dep.tensor_scale(
                f"{prefix}_concat"
            )
        )

        concat_bits = (
            self.dep.tensor_bits(
                f"{prefix}_concat"
            )
        )

        aligned = []

        for result in branches:
            aligned.append(
                requantize_tensor(
                    result["act"],
                    result["scale"],
                    concat_scale,
                    concat_bits,
                )
            )

        aligned.append(
            requantize_tensor(
                pool_act,
                float(
                    pool_cfg[
                        "output_scale"
                    ]
                ),
                concat_scale,
                concat_bits,
            )
        )

        concat = np.concatenate(
            aligned,
            axis=1,
        ).astype(np.int8)

        out_name = (
            f"{prefix}_out_proj"
        )

        out_cfg = self.dep.layer(
            out_name
        )

        out, ms = self._conv(
            out_name,
            concat,
            "acc_to_output",
            output_bits=int(
                out_cfg[
                    "output_bits"
                ]
            ),
        )

        fpga_ms += ms

        shortcut_name = (
            f"{prefix}_shortcut"
        )

        try:
            shortcut_cfg = (
                self.dep.layer(
                    shortcut_name
                )
            )

            shortcut, ms = self._conv(
                shortcut_name,
                x,
                "acc_to_output",
                output_bits=int(
                    shortcut_cfg[
                        "output_bits"
                    ]
                ),
            )

            shortcut_scale = float(
                shortcut_cfg[
                    "output_scale"
                ]
            )

            fpga_ms += ms

        except KeyError:
            shortcut = x
            shortcut_scale = float(
                input_scale
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
            main_q=out,
            main_scale=float(
                out_cfg["output_scale"]
            ),
            shortcut_q=shortcut,
            shortcut_scale=
                shortcut_scale,
            output_scale=add_scale,
            output_bits=add_bits,
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
            output_bits=act_bits,
        )

        if downsample:
            pooled = maxpool1d_2_s2(
                act
            )
        else:
            pooled = None

        return {
            "act": act,
            "pooled": pooled,
            "fpga_ms": fpga_ms,
        }

    def infer(
        self,
        normalized_signal,
    ):
        start_total = (
            time.perf_counter()
        )

        normalized_signal = (
            np.asarray(
                normalized_signal,
                dtype=np.float32,
            )
            .reshape(
                320,
                1,
            )
        )

        stem_cfg = self.dep.layer(
            "stem_conv"
        )

        x = quantize(
            normalized_signal,
            float(
                stem_cfg[
                    "input_scale"
                ]
            ),
            8,
        )

        stem_preact_bits = (
            self.dep.tensor_bits(
                stem_cfg["bn_name"]
            )
            if stem_cfg.get(
                "bn_name"
            )
            else 8
        )

        stem_preact, stem_ms = (
            self._conv(
                "stem_conv",
                x,
                "acc_to_preact",
                output_bits=
                    stem_preact_bits,
            )
        )

        stem_act = gelu_requant(
            q_in=stem_preact,
            input_scale=float(
                stem_cfg[
                    "preactivation_scale"
                ]
            ),
            output_scale=float(
                stem_cfg[
                    "output_scale"
                ]
            ),
            output_bits=int(
                stem_cfg[
                    "output_bits"
                ]
            ),
        )

        fpga_ms = stem_ms

        b1 = self._block(
            "block1",
            stem_act,
            float(
                stem_cfg[
                    "output_scale"
                ]
            ),
            True,
        )

        fpga_ms += b1[
            "fpga_ms"
        ]

        pool1 = b1[
            "pooled"
        ]

        b2 = self._block(
            "block2",
            pool1,
            self.dep.tensor_scale(
                "pool1"
            ),
            True,
        )

        fpga_ms += b2[
            "fpga_ms"
        ]

        pool2 = b2[
            "pooled"
        ]

        b3 = self._block(
            "block3",
            pool2,
            self.dep.tensor_scale(
                "pool2"
            ),
            True,
        )

        fpga_ms += b3[
            "fpga_ms"
        ]

        pool3 = b3[
            "pooled"
        ]

        b4 = self._block(
            "block4",
            pool3,
            self.dep.tensor_scale(
                "pool3"
            ),
            False,
        )

        fpga_ms += b4[
            "fpga_ms"
        ]

        block4_act = b4[
            "act"
        ]

        block4_scale = (
            self.dep.tensor_scale(
                "block4_act"
            )
        )

        gap = global_average_requant(
            block4_act,
            block4_scale,
            self.dep.tensor_scale(
                "gap"
            ),
            self.dep.tensor_bits(
                "gap"
            ),
        )

        gmp = global_max_requant(
            block4_act,
            block4_scale,
            self.dep.tensor_scale(
                "gmp"
            ),
            self.dep.tensor_bits(
                "gmp"
            ),
        )

        head_scale = (
            self.dep.tensor_scale(
                "head_concat"
            )
        )

        head_bits = (
            self.dep.tensor_bits(
                "head_concat"
            )
        )

        head_concat = np.concatenate(
            [
                requantize_tensor(
                    gap,
                    self.dep.tensor_scale(
                        "gap"
                    ),
                    head_scale,
                    head_bits,
                ),

                requantize_tensor(
                    gmp,
                    self.dep.tensor_scale(
                        "gmp"
                    ),
                    head_scale,
                    head_bits,
                ),
            ]
        ).astype(np.int8)

        head_cfg = self.dep.layer(
            "head_dense"
        )

        head_kernel = (
            self.dep.load_array(
                head_cfg[
                    "kernel_file"
                ],
                np.int8,
            )
            .reshape(
                1,
                56,
                24,
            )
        )

        head_bias = (
            self.dep.load_array(
                head_cfg[
                    "bias_file"
                ],
                np.int32,
            )
        )

        head_req = head_cfg[
            "acc_to_preact"
        ]

        head_preact, ms = (
            self.fpga.execute(
                x=head_concat.reshape(
                    1,
                    56,
                ),
                weights=head_kernel,
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
                    self.dep.tensor_bits(
                        "head_bn"
                    ),
                dilation=1,
                depthwise=False,
                weight_bits=int(
                    head_cfg[
                        "bits"
                    ]
                ),
            )
        )

        fpga_ms += ms

        head_act = gelu_requant(
            q_in=head_preact.reshape(
                24,
            ),
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
            ),
        )

        class_cfg = self.dep.layer(
            "class_output"
        )

        class_kernel = (
            self.dep.load_array(
                class_cfg[
                    "kernel_file"
                ],
                np.int8,
            )
        )

        class_bias = (
            self.dep.load_array(
                class_cfg[
                    "bias_file"
                ],
                np.int32,
            )
        )

        # class_output là INT32_LOGITS.
        # Accelerator hiện tại requantize output,
        # nên Dense 24->5 cuối được tính trên ARM.
        logits_int32 = dense_logits_int32(
            head_act,
            class_kernel,
            class_bias,
        )

        logits_float = (
            logits_int32.astype(
                np.float64
            )
            * float(
                class_cfg[
                    "logit_scale"
                ]
            )
        )

        probabilities = softmax_float(
            logits_float
        )

        predicted_id = int(
            np.argmax(
                logits_int32
            )
        )

        total_ms = (
            time.perf_counter()
            - start_total
        ) * 1000.0

        return {
            "source":
                "PYNQ_FPGA_HYBRID",

            "predicted_id":
                predicted_id,

            "predicted_label":
                LABELS[
                    predicted_id
                ],

            "logits_int32":
                [
                    int(v)
                    for v in logits_int32
                ],

            "logits_float":
                [
                    float(v)
                    for v in logits_float
                ],

            "probabilities":
                [
                    float(v)
                    for v in probabilities
                ],

            "timing": {
                "fpga_weighted_ms":
                    float(fpga_ms),

                "total_hybrid_ms":
                    float(total_ms),
            },

            "trace": {
                "input":
                    [320, 1],

                "pool1":
                    list(
                        pool1.shape
                    ),

                "pool2":
                    list(
                        pool2.shape
                    ),

                "pool3":
                    list(
                        pool3.shape
                    ),

                "block4":
                    list(
                        block4_act.shape
                    ),

                "head":
                    list(
                        head_act.shape
                    ),
            },
        }
