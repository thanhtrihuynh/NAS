import sys
from pathlib import Path

import numpy as np


LABELS = ["N", "L", "R", "V", "A"]


class InferenceService:
    def __init__(self, project_root):
        self.root = Path(project_root)
        server_dir = self.root / "server"

        if str(server_dir) not in sys.path:
            sys.path.insert(0, str(server_dir))

        from deployment_loader import DeploymentLoader
        from runtime_to_pool1 import RuntimeToPool1
        from model_block import ResidualInceptionBlock
        from integer_ops import (
            conv1d_same,
            dense_integer,
            dense_logits_int32,
            gelu_requant,
            global_average_requant,
            global_max_requant,
            quantize,
            requantize_tensor,
            softmax_float,
        )

        self.DeploymentLoader = DeploymentLoader
        self.RuntimeToPool1 = RuntimeToPool1
        self.ResidualInceptionBlock = ResidualInceptionBlock
        self.conv1d_same = conv1d_same
        self.dense_integer = dense_integer
        self.dense_logits_int32 = dense_logits_int32
        self.gelu_requant = gelu_requant
        self.global_average_requant = global_average_requant
        self.global_max_requant = global_max_requant
        self.quantize = quantize
        self.requantize_tensor = requantize_tensor
        self.softmax_float = softmax_float
        self.dep = DeploymentLoader(self.root)

    def run(self, normalized_signal):
        normalized_signal = np.asarray(normalized_signal, dtype=np.float32).reshape(320, 1)

        stem_cfg = self.dep.layer("stem_conv")
        ecg_input = self.quantize(
            normalized_signal,
            float(stem_cfg["input_scale"]),
            8,
        )

        stem_kernel = self.dep.load_array(stem_cfg["kernel_file"], np.int8)
        stem_bias = self.dep.load_array(stem_cfg["bias_file"], np.int32)
        stem_req = stem_cfg["acc_to_preact"]

        stem_preact = self.conv1d_same(
            x=ecg_input,
            kernel=stem_kernel,
            bias=stem_bias,
            multiplier=int(stem_req["multiplier"]),
            shift=int(stem_req["shift"]),
            output_bits=8,
            dilation=int(stem_cfg["dilation"]),
        )

        block1_runtime = self.RuntimeToPool1(self.root)
        block1 = block1_runtime.run(stem_preact)
        pool1 = block1["pool1"]

        block_runtime = self.ResidualInceptionBlock(self.dep)

        block2 = block_runtime.run(
            prefix="block2",
            x=pool1,
            input_scale=self.dep.tensor_scale("pool1"),
            downsample=True,
        )
        pool2 = block2["pooled"]

        block3 = block_runtime.run(
            prefix="block3",
            x=pool2,
            input_scale=self.dep.tensor_scale("pool2"),
            downsample=True,
        )
        pool3 = block3["pooled"]

        block4 = block_runtime.run(
            prefix="block4",
            x=pool3,
            input_scale=self.dep.tensor_scale("pool3"),
            downsample=False,
        )
        block4_act = block4["act"]
        block4_scale = self.dep.tensor_scale("block4_act")

        gap = self.global_average_requant(
            x=block4_act,
            input_scale=block4_scale,
            output_scale=self.dep.tensor_scale("gap"),
            output_bits=self.dep.tensor_bits("gap"),
        )

        gmp = self.global_max_requant(
            x=block4_act,
            input_scale=block4_scale,
            output_scale=self.dep.tensor_scale("gmp"),
            output_bits=self.dep.tensor_bits("gmp"),
        )

        head_concat_scale = self.dep.tensor_scale("head_concat")
        head_concat_bits = self.dep.tensor_bits("head_concat")

        gap_aligned = self.requantize_tensor(
            q=gap,
            input_scale=self.dep.tensor_scale("gap"),
            output_scale=head_concat_scale,
            bits=head_concat_bits,
        )

        gmp_aligned = self.requantize_tensor(
            q=gmp,
            input_scale=self.dep.tensor_scale("gmp"),
            output_scale=head_concat_scale,
            bits=head_concat_bits,
        )

        head_concat = np.concatenate([gap_aligned, gmp_aligned]).astype(np.int8)

        head_cfg = self.dep.layer("head_dense")
        head_kernel = self.dep.load_array(head_cfg["kernel_file"], np.int8)
        head_bias = self.dep.load_array(head_cfg["bias_file"], np.int32)
        head_req = head_cfg["acc_to_preact"]

        head_preact = self.dense_integer(
            x=head_concat,
            kernel=head_kernel,
            bias=head_bias,
            multiplier=int(head_req["multiplier"]),
            shift=int(head_req["shift"]),
            output_bits=self.dep.tensor_bits("head_bn"),
        )

        head_act = self.gelu_requant(
            q_in=head_preact,
            input_scale=float(head_cfg["preactivation_scale"]),
            output_scale=float(head_cfg["output_scale"]),
            output_bits=int(head_cfg["output_bits"]),
        )

        class_cfg = self.dep.layer("class_output")
        class_kernel = self.dep.load_array(class_cfg["kernel_file"], np.int8)
        class_bias = self.dep.load_array(class_cfg["bias_file"], np.int32)

        logits_int32 = self.dense_logits_int32(
            x=head_act,
            kernel=class_kernel,
            bias=class_bias,
        )

        logit_scale = float(class_cfg["logit_scale"])
        logits_float = logits_int32.astype(np.float64) * logit_scale
        probabilities = self.softmax_float(logits_float)
        predicted_id = int(np.argmax(logits_int32))

        return {
            "predicted_id": predicted_id,
            "predicted_label": LABELS[predicted_id],
            "logit_scale": logit_scale,
            "logits_int32": [int(x) for x in logits_int32],
            "logits_float": [float(x) for x in logits_float],
            "probabilities": [float(x) for x in probabilities],
            "trace": {
                "input": list(ecg_input.shape),
                "stem": list(stem_preact.shape),
                "pool1": list(pool1.shape),
                "pool2": list(pool2.shape),
                "pool3": list(pool3.shape),
                "block4": list(block4_act.shape),
                "head": list(head_act.shape),
            },
        }
