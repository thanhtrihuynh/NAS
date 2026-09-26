import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import load_train_val_test


def main():
    data = load_train_val_test(
        use_cache=True,
        augment_train=True,
        seed=42,
    )

    (
        X_train, y_train, y_train_int, train_meta,
        X_val, y_val, y_val_int, val_meta,
        X_test, y_test, y_test_int, test_meta,
    ) = data

    print("\nEXPECTED OLD-PROJECT REFERENCE")
    print("Train augmented expected: (82358, 320, 1)")
    print("Val expected            : (15010, 320, 1)")
    print("Test expected           : (15009, 320, 1)")
    print("Raw train meta expected : 70043")
    print()
    print("Actual train:", X_train.shape)
    print("Actual val  :", X_val.shape)
    print("Actual test :", X_test.shape)
    print("Train meta :", train_meta.shape)
    print("Val meta   :", val_meta.shape)
    print("Test meta  :", test_meta.shape)


if __name__ == "__main__":
    main()
