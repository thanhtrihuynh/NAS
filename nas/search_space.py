import json
from pathlib import Path

from config import BEST_CONFIG_PATH

KERNEL_PATTERNS = [
    (3, 5, 7),
    (3, 5, 9),
    (3, 7, 9),
]

CHANNEL_CHOICES = {
    "block1": [8, 12, 16],
    "block2": [12, 16, 20],
    "block3": [16, 20, 24],
    "block4": [20, 24, 28, 32],
}

PRECISION_CHOICES = [4, 8]

BEST_CONFIG = {
    "block1_kernels": (3, 5, 7),
    "block1_channels": 8,
    "block2_kernels": (3, 5, 9),
    "block2_channels": 12,
    "block3_kernels": (3, 5, 9),
    "block3_channels": 24,
    "block4_kernels": (3, 7, 9),
    "block4_channels": 28,
}


def config_to_jsonable(config):
    result = dict(config)
    for key, value in result.items():
        if isinstance(value, tuple):
            result[key] = list(value)
    return result


def config_from_jsonable(config):
    result = dict(config)
    for key in [
        "block1_kernels",
        "block2_kernels",
        "block3_kernels",
        "block4_kernels",
    ]:
        if key in result:
            result[key] = tuple(result[key])
    return result


def save_config(config, path=BEST_CONFIG_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config_to_jsonable(config), f, indent=2)


def load_config(path=BEST_CONFIG_PATH):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return config_from_jsonable(json.load(f))
