from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_DIR = PROJECT_ROOT / "datasets"

CSV_DIR = DATASETS_DIR
RECORDS_DIR = DATASETS_DIR

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"


BASELINE_MODEL_PATH = (
    ARTIFACTS_DIR
    / "best_optimized_mini_inception.keras"
)

BEST_NAS_MODEL_PATH = (
    ARTIFACTS_DIR
    / "best_nas_fp32.keras"
)

BEST_CONFIG_PATH = (
    ARTIFACTS_DIR
    / "best_config.json"
)


TRAIN_CSV = CSV_DIR / "train.csv"
VAL_CSV = CSV_DIR / "val.csv"
TEST_CSV = CSV_DIR / "test.csv"


LABELS = [
    "N",
    "L",
    "R",
    "V",
    "A"
]

LABEL_TO_ID = {
    "N": 0,
    "L": 1,
    "R": 2,
    "V": 3,
    "A": 4
}

ID_TO_LABEL = {
    0: "N",
    1: "L",
    2: "R",
    3: "V",
    4: "A"
}


SEGMENT_LEN = 320
NUM_CLASSES = 5

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

L2_LAMBDA = 1e-4

DROPOUT_1 = 0.06
DROPOUT_2 = 0.08
DROPOUT_HEAD = 0.25

SEED = 42


for directory in [
    DATASETS_DIR,
    ARTIFACTS_DIR,
    RESULTS_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )