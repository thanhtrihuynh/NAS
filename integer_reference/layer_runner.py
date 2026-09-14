from pathlib import Path

import numpy as np

from integer_reference.ops import (
    integer_conv1d_acc,
    integer_depthwise_conv1d_acc,
    integer_pointwise_conv1d_acc,
    integer_dense_acc,
    requantize_from_scale
)


class IntegerLayerRunner:
    def __init__(
        self,
        deployment_dir,
        manifest
    ):
        self.deployment_dir = Path(
            deployment_dir
        )

        self.manifest = manifest

    def load(
        self,
        filename
    ):
        path = (
            self.deployment_dir
            / filename
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        return np.load(
            path
        )

    def run_conv1d_acc(
        self,
        layer_name,
        x_q
    ):
        info = (
            self.manifest[
                "layers"
            ][layer_name]
        )

        kernel = self.load(
            info[
                "kernel_file"
            ]
        )

        bias = self.load(
            info[
                "bias_file"
            ]
        )

        accumulator = (
            integer_conv1d_acc(
                x_q,
                kernel,
                bias,
                stride=
                    info["stride"],
                dilation=
                    info["dilation"],
                padding=
                    info["padding"]
            )
        )

        return (
            accumulator,
            float(
                info[
                    "accumulator_scale"
                ]
            )
        )

    def run_separable_acc(
        self,
        layer_name,
        x_q
    ):
        info = (
            self.manifest[
                "layers"
            ][layer_name]
        )

        depthwise = self.load(
            info[
                "depthwise_file"
            ]
        )

        pointwise = self.load(
            info[
                "pointwise_file"
            ]
        )

        bias = self.load(
            info[
                "bias_file"
            ]
        )

        depthwise_acc = (
            integer_depthwise_conv1d_acc(
                x_q,
                depthwise,
                stride=
                    info["stride"],
                dilation=
                    info["dilation"],
                padding=
                    info["padding"]
            )
        )

        internal_bits = int(
            info["bits"]
        )

        depthwise_q = (
            requantize_from_scale(
                depthwise_acc,
                info[
                    "depthwise_acc_scale"
                ],
                info[
                    "depthwise_output_scale"
                ],
                internal_bits
            )
        )

        pointwise_acc = (
            integer_pointwise_conv1d_acc(
                depthwise_q,
                pointwise,
                bias
            )
        )

        return (
            pointwise_acc,
            float(
                info[
                    "pointwise_acc_scale"
                ]
            )
        )

    def run_dense_acc(
        self,
        layer_name,
        x_q
    ):
        info = (
            self.manifest[
                "layers"
            ][layer_name]
        )

        kernel = self.load(
            info[
                "kernel_file"
            ]
        )

        bias = self.load(
            info[
                "bias_file"
            ]
        )

        accumulator = (
            integer_dense_acc(
                x_q,
                kernel,
                bias
            )
        )

        return (
            accumulator,
            float(
                info[
                    "accumulator_scale"
                ]
            )
        )

    def run_accumulator(
        self,
        layer_name,
        x_q
    ):
        info = (
            self.manifest[
                "layers"
            ][layer_name]
        )

        layer_type = (
            info["type"]
        )

        if layer_type == "Conv1D":
            return (
                self.run_conv1d_acc(
                    layer_name,
                    x_q
                )
            )

        if (
            layer_type
            == "SeparableConv1D"
        ):
            return (
                self.run_separable_acc(
                    layer_name,
                    x_q
                )
            )

        if layer_type == "Dense":
            return (
                self.run_dense_acc(
                    layer_name,
                    x_q
                )
            )

        raise ValueError(
            "Layer type chưa hỗ trợ: "
            f"{layer_type}"
        )