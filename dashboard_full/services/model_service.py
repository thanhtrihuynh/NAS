import json
from pathlib import Path

import numpy as np


class ModelService:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.final_dir = self.root / "artifacts" / "final_w4a4_p99_9"
        self.model_dir = self.final_dir / "deployment_model"
        self.manifest_file = self.model_dir / "deployment_manifest.json"
        self.scales_file = self.final_dir / "deployment_scales.json"
        self.precision_file = self.final_dir / "best_layer_precision_config.json"

        if not self.manifest_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy deployment manifest: {self.manifest_file}"
            )

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.precision_config = None
        if self.precision_file.exists():
            with open(self.precision_file, "r", encoding="utf-8") as f:
                self.precision_config = json.load(f)

    @staticmethod
    def _shape(cfg, *keys):
        for key in keys:
            value = cfg.get(key)
            if value is not None:
                return value
        return None

    def layer_rows(self):
        rows = []
        for name, cfg in self.manifest.get("layers", {}).items():
            rows.append({
                "name": name,
                "type": cfg.get("type"),
                "bits": int(cfg.get("bits", 8)),
                "input_source": cfg.get("input_source"),
                "kernel_shape": self._shape(
                    cfg, "kernel_shape", "pointwise_shape", "depthwise_shape"
                ),
                "depthwise_shape": cfg.get("depthwise_shape"),
                "pointwise_shape": cfg.get("pointwise_shape"),
                "dilation": cfg.get("dilation"),
                "output_bits": cfg.get("output_bits"),
                "output_mode": cfg.get("output_mode"),
            })
        return rows

    def _weighted_values(self):
        total = 0
        visited = set()

        for cfg in self.manifest.get("layers", {}).values():
            for key in ("kernel_file", "depthwise_file", "pointwise_file", "bias_file"):
                filename = cfg.get(key)
                if not filename or filename in visited:
                    continue
                path = self.model_dir / filename
                if path.exists():
                    total += int(np.load(path).size)
                    visited.add(filename)
        return total

    @staticmethod
    def _sepconv_summary(name, cfg):
        if not cfg:
            return None

        dw_shape = cfg.get("depthwise_shape")
        pw_shape = cfg.get("pointwise_shape")

        return {
            "name": name,
            "type": cfg.get("type", "SepConv1D"),
            "kernel": int(dw_shape[0]) if dw_shape else None,
            "input_channels": int(dw_shape[1]) if dw_shape and len(dw_shape) > 1 else None,
            "output_channels": int(pw_shape[-1]) if pw_shape else None,
            "dilation": int(cfg.get("dilation", 1)),
            "bits": int(cfg.get("bits", 8)),
        }

    def architecture(self):
        layers = self.manifest.get("layers", {})
        stem_cfg = layers.get("stem_conv", {})
        stem_shape = stem_cfg.get("kernel_shape")

        stem = {
            "name": "Stem",
            "kernel": int(stem_shape[0]) if stem_shape else None,
            "input_channels": int(stem_shape[1]) if stem_shape and len(stem_shape) > 1 else None,
            "output_channels": int(stem_shape[2]) if stem_shape and len(stem_shape) > 2 else None,
            "bits": int(stem_cfg.get("bits", 8)),
        }

        blocks = []

        for block_index in range(1, 5):
            prefix = f"block{block_index}"
            branches = []

            for branch in ("b1", "b2", "b3"):
                name = f"{prefix}_{branch}_sepconv"
                summary = self._sepconv_summary(name, layers.get(name))
                if summary:
                    branches.append(summary)

            pool_name = f"{prefix}_pool_proj"
            pool_cfg = layers.get(pool_name)
            pool_branch = None

            if pool_cfg:
                shape = pool_cfg.get("kernel_shape")
                pool_branch = {
                    "name": pool_name,
                    "type": pool_cfg.get("type"),
                    "kernel": int(shape[0]) if shape else 1,
                    "input_channels": int(shape[1]) if shape and len(shape) > 1 else None,
                    "output_channels": int(shape[2]) if shape and len(shape) > 2 else None,
                    "dilation": 1,
                    "bits": int(pool_cfg.get("bits", 8)),
                    "display_name": "Pool projection",
                }

            out_cfg = layers.get(f"{prefix}_out_proj")
            out_channels = None
            if out_cfg:
                shape = out_cfg.get("kernel_shape")
                if shape and len(shape) > 2:
                    out_channels = int(shape[2])

            blocks.append({
                "name": f"Block {block_index}",
                "prefix": prefix,
                "branches": branches,
                "pool_branch": pool_branch,
                "output_channels": out_channels,
                "downsample": block_index < 4,
            })

        head_cfg = layers.get("head_dense", {})
        head_shape = head_cfg.get("kernel_shape")
        class_cfg = layers.get("class_output", {})
        class_shape = class_cfg.get("kernel_shape")

        head = {
            "dense_input": int(head_shape[0]) if head_shape else None,
            "dense_output": int(head_shape[1]) if head_shape and len(head_shape) > 1 else None,
            "classes": int(class_shape[1]) if class_shape and len(class_shape) > 1 else 5,
        }

        return {
            "input_shape": [320, 1],
            "stem": stem,
            "blocks": blocks,
            "head": head,
        }

    def summary(self):
        rows = self.layer_rows()
        int4 = [row["name"] for row in rows if row["bits"] == 4]
        int8 = [row["name"] for row in rows if row["bits"] == 8]

        return {
            "name": "Hardware-Aware Mixed-Precision NAS ECG",
            "deployment": "W4A4 p99.9",
            "classes": ["N", "L", "R", "V", "A"],
            "weighted_layers": len(rows),
            "int4_count": len(int4),
            "int8_count": len(int8),
            "int4_layers": int4,
            "int8_layers": int8,
            "weighted_values": self._weighted_values(),
            "layers": rows,
            "precision_config": self.precision_config,
            "architecture": self.architecture(),
        }
