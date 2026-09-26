from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets"
CACHE_DIR = PROJECT_ROOT / "cache"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

LABELS = ["N", "L", "R", "V", "A"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

SEGMENT_LEN = 320
NUM_CLASSES = 5
RANDOM_SEED = 42

CACHE_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
