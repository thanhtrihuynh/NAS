import tensorflow as tf

from config import BEST_NAS_MODEL_PATH, VAL_CSV, RECORDS_DIR
from data.loader import build_xy, to_keras
from training.evaluate import evaluate_model


def main():
    if not BEST_NAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Chưa có model: {BEST_NAS_MODEL_PATH}\n"
            "Hãy copy best_nas_fp32.keras từ Kaggle vào thư mục artifacts/."
        )

    X_val, y_val, _ = build_xy(VAL_CSV, RECORDS_DIR)
    X_val_keras, y_val_cat = to_keras(X_val, y_val)

    model = tf.keras.models.load_model(
        str(BEST_NAS_MODEL_PATH),
        compile=False,
    )

    print("Model:", model.name)
    print("Parameters:", f"{model.count_params():,}")

    result = evaluate_model(
        model,
        X_val_keras,
        y_val_cat,
        print_report=True,
    )

    print("\nKết quả tham chiếu từ Kaggle:")
    print("Accuracy ≈ 0.9940039973")
    print("Macro F1 ≈ 0.9810518564")

    print("\nKết quả local:")
    print("Accuracy =", result["accuracy"])
    print("Macro F1 =", result["macro_f1"])


if __name__ == "__main__":
    main()
