import json
from pathlib import Path

import numpy as np


class DeploymentLoader:

    def __init__(self, project_root):
        self.root = Path(project_root)

        self.final_dir = (
            self.root
            / "artifacts"
            / "final_w4a4_p99_9"
        )

        self.model_dir = (
            self.final_dir
            / "deployment_model"
        )

        manifest_file = (
            self.model_dir
            / "deployment_manifest.json"
        )

        scales_file = (
            self.final_dir
            / "deployment_scales.json"
        )

        if not manifest_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy manifest: {manifest_file}"
            )

        if not scales_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy scales: {scales_file}"
            )

        with open(
            manifest_file,
            "r",
            encoding="utf-8"
        ) as f:
            self.manifest = json.load(f)

        with open(
            scales_file,
            "r",
            encoding="utf-8"
        ) as f:
            self.scales = json.load(f)

    def layer(self, name):
        return self.manifest["layers"][name]

    def tensor_scale(self, name):
        cfg = self.scales["tensor_scales"][name]

        return float(cfg["scale"])

    def tensor_bits(self, name):
        cfg = self.scales["tensor_scales"][name]

        return int(cfg["bits"])

    def load_array(self, filename, dtype=None):
        path = self.model_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {path}"
            )

        arr = np.load(path)

        if dtype is not None:
            arr = arr.astype(dtype)

        return arr

    def load_conv(self, name):
        cfg = self.layer(name)

        kernel = self.load_array(
            cfg["kernel_file"],
            np.int8
        )

        bias = self.load_array(
            cfg["bias_file"],
            np.int32
        )

        return cfg, kernel, bias

    def load_sepconv(self, name):
        cfg = self.layer(name)

        depthwise = self.load_array(
            cfg["depthwise_file"],
            np.int8
        )

        pointwise = self.load_array(
            cfg["pointwise_file"],
            np.int8
        )

        bias = self.load_array(
            cfg["bias_file"],
            np.int32
        )

        return (
            cfg,
            depthwise,
            pointwise,
            bias
        )