import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_DIR
from models.baseline_model import (
    BASELINE_ARCHITECTURE,
    build_baseline_model,
)
from nas.analytical_cost import estimate_architecture_cost
from nas.cost import estimate_macs
from nas.pruned_search import theoretical_search_space_size


def main():
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("DATASET_DIR  :", DATASET_DIR)
    print("DATASET exists:", DATASET_DIR.exists())

    size = theoretical_search_space_size()
    print("SEARCH SPACE :", size)

    if size != 992256:
        raise RuntimeError("Search space must be 992256.")

    known = {
        "block1_kernels": [3, 7, 9],
        "block1_channels": 4,
        "block2_kernels": [5, 7, 9],
        "block2_channels": 4,
        "block3_kernels": [3, 5, 9],
        "block3_channels": 8,
        "block4_kernels": [3, 5, 7],
        "block4_channels": 36,
    }

    kc = estimate_architecture_cost(known)
    print("KNOWN:", kc.to_dict())

    if (kc.params, kc.macs) != (7373, 340408):
        raise RuntimeError("Analytical estimator check failed.")

    bc = estimate_architecture_cost(
        BASELINE_ARCHITECTURE
    )
    print("BASELINE analytical:", bc.to_dict())

    if (bc.params, bc.macs) != (6473, 432120):
        raise RuntimeError("Baseline cost check failed.")

    model = build_baseline_model()
    keras_params = model.count_params()
    keras_macs = estimate_macs(model)

    print("BASELINE Keras Params:", keras_params)
    print("BASELINE Keras MACs  :", keras_macs)

    if keras_params != bc.params:
        raise RuntimeError("Keras/analytical Params mismatch.")

    if keras_macs != bc.macs:
        raise RuntimeError("Keras/analytical MACs mismatch.")

    print("\nCHECK PASSED")


if __name__ == "__main__":
    main()
