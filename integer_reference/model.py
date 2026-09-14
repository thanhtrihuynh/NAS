import json
from pathlib import Path

import numpy as np

from integer_reference.ops import (
    quantize_tensor
)

from integer_reference.layer_runner import (
    IntegerLayerRunner
)

from integer_reference.block import (
    IntegerResidualBlock,
    get_tensor_info,
    run_weight_layer_output
)

from integer_reference.pooling import (
    max_pool1d_integer,
    global_average_pool1d_integer,
    global_max_pool1d_integer
)

from integer_reference.merge_ops import (
    align_scale,
    concat_integer
)


class IntegerReferenceModel:
    def __init__(
        self,
        deployment_dir
    ):
        self.deployment_dir = Path(
            deployment_dir
        )

        manifest_path = (
            self.deployment_dir
            / "deployment_manifest.json"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                manifest_path
            )

        with manifest_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            self.manifest = (
                json.load(f)
            )

        self.runner = (
            IntegerLayerRunner(
                self.deployment_dir,
                self.manifest
            )
        )

        self.block1 = (
            IntegerResidualBlock(
                1,
                self.runner,
                self.manifest
            )
        )

        self.block2 = (
            IntegerResidualBlock(
                2,
                self.runner,
                self.manifest
            )
        )

        self.block3 = (
            IntegerResidualBlock(
                3,
                self.runner,
                self.manifest
            )
        )

        self.block4 = (
            IntegerResidualBlock(
                4,
                self.runner,
                self.manifest
            )
        )

    def _pool_between_blocks(
        self,
        x_q,
        source_scale,
        pool_name
    ):
        # Kiến trúc:
        # 320 -> 160 -> 80 -> 40

        pooled_q = (
            max_pool1d_integer(
                x_q,
                pool_size=2,
                stride=2,
                padding="valid"
            )
        )

        (
            target_scale,
            target_bits
        ) = get_tensor_info(
            self.manifest,
            pool_name
        )

        pooled_q = (
            align_scale(
                pooled_q,
                source_scale,
                target_scale,
                target_bits
            )
        )

        return (
            pooled_q,
            target_scale,
            target_bits
        )

    def forward(
        self,
        x_float,
        return_trace=False
    ):
        x_float = np.asarray(
            x_float,
            dtype=np.float32
        )

        if x_float.ndim == 2:
            x_float = (
                x_float[
                    :,
                    :,
                    None
                ]
            )

        if x_float.ndim != 3:
            raise ValueError(
                "Input phải có shape "
                "[B, 320, 1]"
            )

        if x_float.shape[-1] != 1:
            raise ValueError(
                "ECG input phải có 1 channel."
            )

        trace = (
            {}
            if return_trace
            else None
        )

        # =========================
        # Input Quantization
        # =========================

        (
            input_scale,
            input_bits
        ) = get_tensor_info(
            self.manifest,
            "__input__"
        )

        x_q = quantize_tensor(
            x_float,
            input_scale,
            input_bits
        )

        if trace is not None:
            trace[
                "__input__"
            ] = {
                "tensor":
                    x_q,
                "scale":
                    input_scale
            }

        # =========================
        # Stem
        # =========================

        (
            x_q,
            x_scale,
            x_bits
        ) = run_weight_layer_output(
            self.runner,
            self.manifest,
            "stem_conv",
            x_q
        )

        if trace is not None:
            trace[
                "stem_act"
            ] = {
                "tensor":
                    x_q,
                "scale":
                    x_scale
            }

        # =========================
        # Block 1
        # =========================

        (
            x_q,
            x_scale,
            x_bits
        ) = self.block1.forward(
            x_q,
            x_scale,
            x_bits,
            trace
        )

        (
            x_q,
            x_scale,
            x_bits
        ) = self._pool_between_blocks(
            x_q,
            x_scale,
            "pool1"
        )

        if trace is not None:
            trace[
                "pool1"
            ] = {
                "tensor":
                    x_q,
                "scale":
                    x_scale
            }

        # =========================
        # Block 2
        # =========================

        (
            x_q,
            x_scale,
            x_bits
        ) = self.block2.forward(
            x_q,
            x_scale,
            x_bits,
            trace
        )

        (
            x_q,
            x_scale,
            x_bits
        ) = self._pool_between_blocks(
            x_q,
            x_scale,
            "pool2"
        )

        if trace is not None:
            trace[
                "pool2"
            ] = {
                "tensor":
                    x_q,
                "scale":
                    x_scale
            }

        # =========================
        # Block 3
        # =========================

        (
            x_q,
            x_scale,
            x_bits
        ) = self.block3.forward(
            x_q,
            x_scale,
            x_bits,
            trace
        )

        (
            x_q,
            x_scale,
            x_bits
        ) = self._pool_between_blocks(
            x_q,
            x_scale,
            "pool3"
        )

        if trace is not None:
            trace[
                "pool3"
            ] = {
                "tensor":
                    x_q,
                "scale":
                    x_scale
            }

        # =========================
        # Block 4
        # =========================

        (
            x_q,
            x_scale,
            x_bits
        ) = self.block4.forward(
            x_q,
            x_scale,
            x_bits,
            trace
        )

        # =========================
        # Global Pooling
        # =========================

        gap_q = (
            global_average_pool1d_integer(
                x_q
            )
        )

        gmp_q = (
            global_max_pool1d_integer(
                x_q
            )
        )

        (
            head_concat_scale,
            head_concat_bits
        ) = get_tensor_info(
            self.manifest,
            "head_concat"
        )

        head_q = (
            concat_integer(
                [
                    gap_q,
                    gmp_q
                ],
                [
                    x_scale,
                    x_scale
                ],
                head_concat_scale,
                head_concat_bits,
                axis=-1
            )
        )

        if trace is not None:
            trace[
                "gap"
            ] = {
                "tensor":
                    gap_q,
                "scale":
                    x_scale
            }

            trace[
                "gmp"
            ] = {
                "tensor":
                    gmp_q,
                "scale":
                    x_scale
            }

            trace[
                "head_concat"
            ] = {
                "tensor":
                    head_q,
                "scale":
                    head_concat_scale
            }

        # =========================
        # Head Dense + GELU
        # =========================

        (
            head_q,
            head_scale,
            head_bits
        ) = run_weight_layer_output(
            self.runner,
            self.manifest,
            "head_dense",
            head_q
        )

        if trace is not None:
            trace[
                "head_act"
            ] = {
                "tensor":
                    head_q,
                "scale":
                    head_scale
            }

        # =========================
        # Final Dense
        # =========================

        (
            logits_int,
            logit_scale
        ) = self.runner.run_dense_acc(
            "class_output",
            head_q
        )

        logits_int = (
            logits_int.astype(
                np.int64
            )
        )

        logits_float = (
            logits_int.astype(
                np.float32
            )
            * float(
                logit_scale
            )
        )

        predictions = np.argmax(
            logits_int,
            axis=-1
        ).astype(
            np.int64
        )

        result = {
            "logits_int":
                logits_int,

            "logit_scale":
                float(
                    logit_scale
                ),

            "logits_float":
                logits_float,

            "predictions":
                predictions
        }

        if return_trace:
            result[
                "trace"
            ] = trace

        return result

    def predict(
        self,
        x_float
    ):
        result = self.forward(
            x_float,
            return_trace=False
        )

        return result[
            "predictions"
        ]