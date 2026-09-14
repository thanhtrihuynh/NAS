import json
from pathlib import Path

import numpy as np


class DeploymentLoader:
    def __init__(
        self,
        deployment_root,
    ):
        self.final_dir = Path(
            deployment_root
        )

        self.model_dir = (
            self.final_dir
            / "deployment_model"
        )

        with open(
            self.model_dir
            / "deployment_manifest.json",
            "r",
            encoding="utf-8",
        ) as f:
            self.manifest = json.load(f)

        scales_file = (
            self.final_dir
            / "deployment_scales.json"
        )

        with open(
            scales_file,
            "r",
            encoding="utf-8",
        ) as f:
            self.scales = json.load(f)

    def layer(self, name):
        return self.manifest[
            "layers"
        ][name]

    def load_array(
        self,
        filename,
        dtype=None,
    ):
        arr = np.load(
            self.model_dir
            / filename
        )

        if dtype is not None:
            arr = arr.astype(
                dtype
            )

        return arr

    def tensor_scale(
        self,
        name,
    ):
        return float(
            self.scales[
                "tensor_scales"
            ][name]["scale"]
        )

    def tensor_bits(
        self,
        name,
    ):
        return int(
            self.scales[
                "tensor_scales"
            ][name]["bits"]
        )
