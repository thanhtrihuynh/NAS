from models.nas_model import build_nas_model

BASELINE_ARCHITECTURE = {
    "block1_kernels": [3, 5, 7],
    "block1_channels": 8,
    "block2_kernels": [3, 5, 7],
    "block2_channels": 12,
    "block3_kernels": [3, 5, 7],
    "block3_channels": 16,
    "block4_kernels": [3, 5, 7],
    "block4_channels": 20,
}


def build_baseline_model():
    return build_nas_model(BASELINE_ARCHITECTURE)
